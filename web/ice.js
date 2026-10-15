// ICE (STUN/TURN) servers for every page that opens a WebRTC session.
// The list comes from the server so relay credentials never live in page source.
// Without a relay, a viewer behind a VPN or a strict firewall gets no video.
var ICE_FALLBACK = [{ urls: ['stun:stun.cloudflare.com:3478'] }];

async function fetchIceServers() {
    try {
        const res = await fetch('/api/ice', { cache: 'no-store' });
        const body = await res.json();
        const servers = body && body.data && body.data.iceServers;
        if (body.code === 0 && Array.isArray(servers) && servers.length) {
            return servers;
        }
        console.warn('GET /api/ice returned no ICE servers, using STUN only:', body);
    } catch (e) {
        console.warn('GET /api/ice failed, using STUN only:', e);
    }
    return ICE_FALLBACK;
}

// Resolve once ICE gathering has finished, or after `maxMs` with whatever was
// gathered so far: an unreachable relay URL must not stall the offer.
function waitForIceGathering(pc, maxMs) {
    return new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
            resolve();
            return;
        }
        const finish = () => {
            clearTimeout(timer);
            pc.removeEventListener('icegatheringstatechange', check);
            resolve();
        };
        const check = () => {
            if (pc.iceGatheringState === 'complete') finish();
        };
        const timer = setTimeout(finish, maxMs);
        pc.addEventListener('icegatheringstatechange', check);
    });
}
