var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete (bounded: see waitForIceGathering)
        return waitForIceGathering(pc, 4000);
    }).then(() => {
        var offer = pc.localDescription;
        var body = { sdp: offer.sdp, type: offer.type };
        var avatarEl = document.getElementById('avatar-select');
        if (avatarEl && avatarEl.value) {
            body.avatar = avatarEl.value;
        }
        return fetch('/offer', {
            body: JSON.stringify(body),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }
        return response.json();
    }).then((answer) => {
        // the server answers {code: -1, msg} instead of an SDP when it refuses (e.g. session limit)
        if (answer.code === -1 || !answer.sessionid) {
            throw new Error(answer.msg || 'server refused the session');
        }
        document.getElementById('sessionid').value = answer.sessionid;
        if (window.onWebRTCSession) window.onWebRTCSession(answer.sessionid);
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        if (window.onWebRTCError) {
            window.onWebRTCError(e);
        } else {
            alert(e);
        }
    });
}

async function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = await fetchIceServers();
    }

    pc = new RTCPeerConnection(config);

    // report the real connection state to the page
    pc.addEventListener('connectionstatechange', () => {
        const state = pc.connectionState;
        if (state === 'connected' && window.onWebRTCConnected) {
            window.onWebRTCConnected();
        } else if (state === 'disconnected' && window.onWebRTCUnstable) {
            window.onWebRTCUnstable();
        } else if ((state === 'failed' || state === 'closed') && window.onWebRTCDisconnected) {
            window.onWebRTCDisconnected(state);
        }
    });

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        if (pc) pc.close();
    }, 500);
}

window.addEventListener('pagehide', function() {
    if (pc) pc.close();
});

// Ask before leaving only while a session is live or being set up. Asking
// unconditionally made every refresh (including the one used to recover from a
// failed connection) show a confirmation dialog.
window.onbeforeunload = function (e) {
    if (!pc || (pc.connectionState !== 'connected' && pc.connectionState !== 'connecting')) {
        return undefined;
    }
    e.preventDefault();
    e.returnValue = '';
    return '';
};
