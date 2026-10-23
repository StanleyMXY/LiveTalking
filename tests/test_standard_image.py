import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from server import avatar_routes, standard_image
from server.standard_image import (
    BAD_PHOTO, GENERATION_FAILED, NO_FACE, ImageEditor, StandardImageError, StandardImageJobs,
    build_prompt, fit_for_edit, generate_standard_image, output_size,
)


def photo(path, w=800, h=1000):
    cv2.imwrite(path, (np.random.rand(h, w, 3) * 255).astype('uint8'))
    return path


def png_bytes(w=64, h=64):
    ok, buf = cv2.imencode('.png', (np.random.rand(h, w, 3) * 255).astype('uint8'))
    return buf.tobytes()


class PromptTest(unittest.TestCase):
    def test_follows_the_products_realistic_prompt(self):
        prompt = build_prompt()
        for phrase in ("photorealistic portrait of the same person", "down to the waist",
                       "looking into the camera", "upper front teeth are clearly visible",
                       "preserve every facial feature exactly", standard_image.DEFAULT_OUTFIT,
                       standard_image.DEFAULT_BACKGROUND):
            self.assertIn(phrase, prompt)
        self.assertNotIn("cartoon", prompt.lower().replace("not a cartoon", ""))
        self.assertIn("cartoon", standard_image.NEGATIVE_PROMPT)  # the demo never produces a cartoon

    def test_braces_in_the_texts_are_not_interpreted(self):
        self.assertIn("a {red} coat", build_prompt(outfit="a {red} coat"))


class SizeTest(unittest.TestCase):
    def test_output_size_matches_the_documented_examples(self):
        self.assertEqual(output_size(1080, 1350), (912, 1152))
        self.assertEqual(output_size(1080, 1620), (832, 1248))

    def test_large_photo_is_scaled_to_2048(self):
        uri, w, h = fit_for_edit(np.zeros((4000, 3000, 3), np.uint8))
        self.assertEqual(max(w, h), 2048)
        self.assertTrue(uri.startswith("data:image/"))

    def test_small_photo_is_scaled_up_to_384(self):
        _, w, h = fit_for_edit(np.zeros((300, 200, 3), np.uint8))
        self.assertGreaterEqual(min(w, h), 384)

    def test_extreme_proportions_are_refused(self):
        with self.assertRaises(StandardImageError) as ctx:
            fit_for_edit(np.zeros((3000, 200, 3), np.uint8))
        self.assertEqual(ctx.exception.code, BAD_PHOTO)


class EditorTest(unittest.TestCase):
    def response(self, status=200, body=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = body if body is not None else {}
        return r

    def ok_body(self):
        return {"output": {"choices": [{"message": {"content": [{"image": "https://cdn.example/result.png"}]}}]}}

    def download(self):
        r = MagicMock()
        r.status_code = 200
        r.content = png_bytes()
        return r

    def editor(self):
        return ImageEditor("key", "https://dashscope-intl.example/api/v1", "qwen-image-3.0", timeout_s=5)

    def test_request_has_the_products_shape_and_the_result_is_downloaded(self):
        with patch("server.standard_image.requests.post", return_value=self.response(200, self.ok_body())) as post, \
                patch("server.standard_image.requests.get", return_value=self.download()):
            img = self.editor().edit("data:image/png;base64,AAAA", 1080, 1350, "PROMPT", "NEG", 123)
        self.assertEqual(img.shape[:2], (64, 64))
        (url,), kwargs = post.call_args
        self.assertEqual(url, "https://dashscope-intl.example/api/v1/services/aigc/multimodal-generation/generation")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer key")
        body = kwargs["json"]
        self.assertEqual(body["model"], "qwen-image-3.0")
        self.assertEqual(body["input"]["messages"][0]["content"], [{"image": "data:image/png;base64,AAAA"}, {"text": "PROMPT"}])
        self.assertEqual(body["parameters"], {"n": 1, "watermark": False, "negative_prompt": "NEG",
                                              "prompt_extend": False, "seed": 123, "size": "912*1152"})

    def test_throttling_is_retried(self):
        with patch("server.standard_image.requests.post",
                   side_effect=[self.response(429), self.response(200, self.ok_body())]) as post, \
                patch("server.standard_image.requests.get", return_value=self.download()), \
                patch("server.standard_image.time.sleep"):
            self.editor().edit("u", 100, 100, "p", "n", 1)
        self.assertEqual(post.call_count, 2)

    def test_connect_timeouts_are_retried_then_reported(self):
        with patch("server.standard_image.requests.post", side_effect=requests.ConnectTimeout()) as post, \
                patch("server.standard_image.time.sleep"):
            with self.assertRaises(StandardImageError):
                self.editor().edit("u", 100, 100, "p", "n", 1)
        self.assertEqual(post.call_count, 3)

    def test_a_read_timeout_is_not_repeated_because_the_edit_may_be_billed(self):
        with patch("server.standard_image.requests.post", side_effect=requests.ReadTimeout()) as post:
            with self.assertRaises(StandardImageError):
                self.editor().edit("u", 100, 100, "p", "n", 1)
        self.assertEqual(post.call_count, 1)

    def test_content_safety_rejection_is_explained(self):
        body = {"code": "DataInspectionFailed", "message": "Input data may contain inappropriate content."}
        with patch("server.standard_image.requests.post", return_value=self.response(400, body)):
            with self.assertRaises(StandardImageError) as ctx:
                self.editor().edit("u", 100, 100, "p", "n", 1)
        self.assertIn("content safety", str(ctx.exception))


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = photo(os.path.join(self.tmp.name, "in.jpg"))

    def editor(self):
        editor = MagicMock()
        editor.edit.side_effect = lambda *a: (np.random.rand(96, 96, 3) * 255).astype('uint8')
        return editor

    def test_photo_without_a_face_is_refused_before_any_paid_call(self):
        editor = self.editor()
        with self.assertRaises(StandardImageError) as ctx:
            generate_standard_image(self.path, editor, face_check=lambda img: False)
        self.assertEqual(ctx.exception.code, NO_FACE)
        editor.edit.assert_not_called()

    def test_unreadable_file_is_reported(self):
        bad = os.path.join(self.tmp.name, "bad.jpg")
        open(bad, "wb").write(b"not an image")
        with self.assertRaises(StandardImageError) as ctx:
            generate_standard_image(bad, self.editor(), face_check=lambda img: True)
        self.assertEqual(ctx.exception.code, BAD_PHOTO)

    def test_an_attempt_without_a_face_is_retried_with_a_new_seed(self):
        editor = self.editor()
        # photo: yes, 1st result: no face, 2nd result: face
        answers = iter([True, False, True])
        png = generate_standard_image(self.path, editor, face_check=lambda img: next(answers))
        self.assertEqual(editor.edit.call_count, 2)
        seeds = [call.args[-1] for call in editor.edit.call_args_list]
        self.assertNotEqual(seeds[0], seeds[1])
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR))

    def test_gives_up_after_the_last_attempt(self):
        editor = self.editor()
        answers = iter([True] + [False] * 3)
        with self.assertRaises(StandardImageError) as ctx:
            generate_standard_image(self.path, editor, face_check=lambda img: next(answers), max_attempts=3)
        self.assertEqual(ctx.exception.code, GENERATION_FAILED)
        self.assertEqual(editor.edit.call_count, 3)


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def wait(self, jobs, job_id, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            job = jobs.get(job_id)
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_job_succeeds_stores_the_image_and_drops_the_upload(self):
        upload = photo(os.path.join(self.tmp.name, "up.jpg"))
        editor = MagicMock()
        editor.edit.return_value = (np.random.rand(64, 64, 3) * 255).astype('uint8')
        jobs = StandardImageJobs(self.tmp.name, lambda: editor, face_check=lambda img: True)
        job = self.wait(jobs, jobs.start(upload))
        self.assertEqual(job["status"], "done")
        self.assertTrue(os.path.isfile(job["path"]))
        self.assertFalse(os.path.exists(upload))

    def test_job_failure_carries_the_user_message_and_code(self):
        upload = photo(os.path.join(self.tmp.name, "up.jpg"))
        jobs = StandardImageJobs(self.tmp.name, lambda: MagicMock(), face_check=lambda img: False)
        job = self.wait(jobs, jobs.start(upload))
        self.assertEqual((job["status"], job["error_code"]), ("failed", NO_FACE))
        self.assertIn("No face", job["error"])

    def test_missing_key_is_a_clear_failure(self):
        upload = photo(os.path.join(self.tmp.name, "up.jpg"))
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
            jobs = StandardImageJobs(self.tmp.name, standard_image.editor_from_environment, face_check=lambda img: True)
            job = self.wait(jobs, jobs.start(upload))
        self.assertEqual(job["error_code"], standard_image.NOT_CONFIGURED)

    def test_unknown_job_is_none(self):
        self.assertIsNone(StandardImageJobs(self.tmp.name, lambda: None).get("0" * 32))


class RoutesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        editor = MagicMock()
        editor.edit.return_value = (np.random.rand(64, 64, 3) * 255).astype('uint8')
        self.jobs = StandardImageJobs(self.tmp.name, lambda: editor, face_check=lambda img: True)
        patcher = patch.object(avatar_routes, "_standard_jobs", self.jobs)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.makedirs("./data/tmp", exist_ok=True)
        app = web.Application()
        avatar_routes.setup_avatar_routes(app)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def upload(self, name="me.jpg", data=None):
        form = FormData()
        form.add_field("photo", data or png_bytes(200, 200), filename=name, content_type="image/jpeg")
        return form

    async def finish(self, job_id):
        for _ in range(400):
            body = await (await self.client.get(f"/api/avatar/standard/{job_id}")).json()
            if body["data"]["status"] != "running":
                return body["data"]
            await asyncio.sleep(0.02)
        self.fail("job did not finish")

    async def test_full_flow_generate_preview_use(self):
        body = await (await self.client.post("/api/avatar/standard", data=self.upload())).json()
        self.assertEqual(body["code"], 0)
        job_id = body["data"]["job_id"]
        status = await self.finish(job_id)
        self.assertEqual(status["status"], "done")

        image = await self.client.get(f"/api/avatar/standard/{job_id}/image")
        self.assertEqual(image.status, 200)
        self.assertTrue((await image.read()).startswith(b"\x89PNG"))

        with patch.object(avatar_routes.task_manager, "add_task", return_value="task-1") as add_task:
            used = await (await self.client.post(f"/api/avatar/standard/{job_id}/use", json={"avatar_id": "my_look"})).json()
        self.assertEqual(used["data"]["task_id"], "task-1")
        args = add_task.call_args.args
        self.assertEqual(args[0:2], ("wav2lip", "my_look"))
        self.assertEqual(args[2]["video_path"], self.jobs.get(job_id)["path"])
        self.assertEqual(args[2]["pads"], [0, 10, 0, 0])   # same defaults as an uploaded file

    async def test_use_rejects_a_bad_avatar_id_and_an_unfinished_image(self):
        job_id = (await (await self.client.post("/api/avatar/standard", data=self.upload())).json())["data"]["job_id"]
        await self.finish(job_id)
        bad = await (await self.client.post(f"/api/avatar/standard/{job_id}/use", json={"avatar_id": "../x"})).json()
        self.assertEqual(bad["code"], -1)
        self.jobs._jobs[job_id]["status"] = "running"
        early = await (await self.client.post(f"/api/avatar/standard/{job_id}/use", json={"avatar_id": "ok"})).json()
        self.assertIn("not ready", early["msg"])

    async def test_non_image_upload_and_unknown_job(self):
        bad = await (await self.client.post("/api/avatar/standard", data=self.upload("note.txt", b"text"))).json()
        self.assertEqual(bad["code"], -1)
        for url in ("/api/avatar/standard/" + "0" * 32, "/api/avatar/standard/not-a-job", "/api/avatar/standard/" + "0" * 32 + "/image"):
            self.assertEqual((await (await self.client.get(url)).json())["code"], 404)


if __name__ == "__main__":
    unittest.main()
