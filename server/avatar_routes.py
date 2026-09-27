import os
import json
import re
import uuid
from aiohttp import web
from server.task_manager import task_manager
from utils.logger import logger

_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IMAGE_TO_VIDEO_FRAMES = 120

def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )

def json_error(msg: str, code: int = -1):
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )

def _image_to_video(image_path: str) -> str:
    """把一张静态图片重复写成一个短视频，供 genavatar.py 的视频输入管线复用。"""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"failed to read image: {image_path}")
    h, w = img.shape[:2]

    synth_path = image_path + "_synth.mp4"
    writer = cv2.VideoWriter(synth_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    for _ in range(IMAGE_TO_VIDEO_FRAMES):
        writer.write(img)
    writer.release()
    return synth_path


async def create_avatar_task(request):
    """
    POST /api/avatar/task
    Parameters: model, avatar_id, video_file (upload), video_path (local), ...
    """
    try:
        if request.content_type == 'multipart/form-data':
            reader = await request.multipart()
            params = {}
            video_path = None

            while True:
                part = await reader.next()
                if part is None:
                    break

                if part.name == 'video_file':
                    filename = part.filename
                    temp_dir = os.path.abspath('./data/tmp')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                    with open(video_path, 'wb') as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    value = await part.text()
                    params[part.name] = value

            if video_path:
                params['video_path'] = video_path
        else:
            params = await request.json()

        model_type = params.get('model')
        avatar_id = params.get('avatar_id')

        if not model_type or not avatar_id:
            return json_error("model and avatar_id are required")
        if not _SAFE_ID_RE.match(avatar_id):
            return json_error("avatar_id must match ^[A-Za-z0-9_-]{1,64}$")

        if 'video_path' not in params:
            return json_error("video_file or video_path is required")

        data_path = './data/avatars'

        video_path = params['video_path']
        if not os.path.isabs(video_path):
            video_path = os.path.join(data_path, video_path)

        ext = os.path.splitext(video_path)[1].lower()
        if ext in IMAGE_EXTS:
            video_path = _image_to_video(video_path)

        save_path = data_path

        task_params = {
            "video_path": video_path,
            "save_path": save_path,
            "img_size": int(params.get('img_size', 256)),
            "nosmooth": params.get('nosmooth', 'false').lower() == 'true' if isinstance(params.get('nosmooth'), str) else params.get('nosmooth', False),
            "bbox_shift": int(params.get('bbox_shift', 0)),
            "extra_margin": int(params.get('extra_margin', 10)),
            "parsing_mode": params.get('parsing_mode', 'jaw'),
            "version": params.get('version', 'v15'),
            "face_det_batch_size": int(params.get('face_det_batch_size', 16))
        }

        pads_str = params.get('pads', "0 10 0 0")
        if isinstance(pads_str, str):
            task_params['pads'] = [int(x) for x in pads_str.split()]
        else:
            task_params['pads'] = pads_str

        task_id_input = params.get('task_id')
        notify_url = params.get('notifyurl')

        task_id = task_manager.add_task(model_type, avatar_id, task_params, task_id=task_id_input, notify_url=notify_url)
        return json_ok(data={"task_id": task_id})

    except Exception as e:
        logger.exception("create_avatar_task error:")
        return json_error(str(e))

async def get_avatar_task_status(request):
    """
    GET /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    task = task_manager.get_task(task_id)
    if not task:
        return json_error("Task not found", code=404)

    return json_ok(data=task.to_dict())

async def list_avatar_tasks(request):
    """
    GET /api/avatar/tasks
    """
    tasks = task_manager.list_tasks()
    return json_ok(data={"tasks": tasks})

async def delete_avatar_task(request):
    """
    DELETE /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    success, msg = task_manager.delete_task(task_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"msg": msg})

async def list_avatars(request):
    """
    GET /api/avatar/list — 扫描 data/avatars/ 下已经准备好的形象（含 coords.pkl 的目录）
    """
    data_path = os.path.abspath('./data/avatars')
    avatars = []
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            avatar_dir = os.path.join(data_path, name)
            coords_path = os.path.join(avatar_dir, 'coords.pkl')
            if os.path.isdir(avatar_dir) and os.path.isfile(coords_path):
                full_imgs_dir = os.path.join(avatar_dir, 'full_imgs')
                frame_count = len(os.listdir(full_imgs_dir)) if os.path.isdir(full_imgs_dir) else 0
                avatars.append({"id": name, "frame_count": frame_count})
    return json_ok(data={"avatars": avatars})


def setup_avatar_routes(app):
    app.router.add_post("/api/avatar/task", create_avatar_task)
    app.router.add_get("/api/avatar/task/{task_id}", get_avatar_task_status)
    app.router.add_delete("/api/avatar/task/{task_id}", delete_avatar_task)
    app.router.add_get("/api/avatar/tasks", list_avatar_tasks)
    app.router.add_get("/api/avatar/list", list_avatars)
