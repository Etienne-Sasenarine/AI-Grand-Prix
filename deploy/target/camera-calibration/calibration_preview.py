#!/usr/bin/env python3
"""Live calibration with numbered outlines of saved checkerboard poses."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import calibrate_camera as calibration

calibration.cv2, calibration.np = cv2, np
cv2.setNumThreads(2)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', default='http://127.0.0.1:8080/stream')
p.add_argument('--previous', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--port', type=int, default=8081)
p.add_argument('--cols', type=int, default=7)
p.add_argument('--rows', type=int, default=10)
p.add_argument('--square-mm', type=float, default=25)
p.add_argument('--views', type=int, default=25)
p.add_argument('--seconds', type=int, default=600)
a = p.parse_args()
if min(a.cols, a.rows) < 3 or not np.isfinite(a.square_mm) or a.square_mm <= 0 or a.views < 12 or a.seconds <= 0:
    p.error('Invalid board dimensions, target views, or duration')
a.output.mkdir(parents=True, exist_ok=False)
(a.output / 'captures').mkdir()
lock = threading.Lock()
state = dict(frame=None, frame_time=0., sequence=0, points=[], names=[], size=None,
             current=None, current_time=0., status='Starting camera', count=0,
             phase='starting', rms=None, output=str(a.output))


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = min(1., 960 / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale)
    found, corners = cv2.findChessboardCornersSB(small, (a.cols, a.rows),
                                                flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    corners = (corners / scale).astype(np.float32)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001))


def add(frame, corners):
    size = (frame.shape[1], frame.shape[0])
    with lock:
        old_size, previous = state['size'], list(state['points'])
    if old_size is not None and size != old_size:
        raise ValueError('Image size changed; calibration requires a consistent camera configuration')
    for old in previous:
        difference = min(np.linalg.norm(corners-old, axis=2).mean(),
                         np.linalg.norm(corners[::-1]-old, axis=2).mean())
        if difference / np.hypot(*size) < .015:
            return False
    path = a.output / 'captures' / f'frame_{len(previous):03d}.png'
    if not cv2.imwrite(str(path), frame):
        raise OSError(f'Unable to save {path}')
    with lock:
        state['size'] = size
        state['points'].append(corners.copy())
        state['names'].append(str(path))
        state['count'] = len(state['points'])
    print(f'Accepted {len(previous)+1}: {path}', flush=True)
    return True


def reader():
    cap = cv2.VideoCapture(a.source)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open live stream')
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Live stream stopped')
            with lock:
                state.update(frame=frame, frame_time=time.monotonic(), sequence=state['sequence']+1)
    except Exception as exc:
        with lock:
            state.update(status=str(exc), phase='error')
        print(str(exc), flush=True)
    finally:
        cap.release()


def worker():
    try:
        if a.previous:
            for path in sorted(a.previous.glob('*.png')):
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                corners = detect(frame)
                if corners is not None:
                    add(frame, corners)
        with lock:
            if state['phase'] == 'error':
                return
            state.update(phase='collecting', status='Show the whole checkerboard; hold still')
        start = time.monotonic()
        last_saved, last_sequence = 0., -1
        last_corners, last_detection = None, 0.
        while time.monotonic()-start < a.seconds:
            with lock:
                frame, stamp, seq = state['frame'], state['frame_time'], state['sequence']
                if state['phase'] == 'error':
                    return
                if state['count'] >= a.views:
                    break
            if frame is None or seq == last_sequence:
                time.sleep(.05)
                continue
            if time.monotonic()-stamp > 3:
                raise RuntimeError('Camera stream is stale')
            last_sequence = seq
            corners = detect(frame)
            now = time.monotonic()
            if corners is None:
                last_corners = None
                message = 'Board not detected: show every inner corner'
            else:
                stable = (last_corners is not None and now-last_detection < 2 and
                          min(np.linalg.norm(corners-last_corners, axis=2).mean(),
                              np.linalg.norm(corners[::-1]-last_corners, axis=2).mean()) < 5)
                message = 'Board detected: hold still'
                if stable and now-last_saved >= 3:
                    if add(frame, corners):
                        last_saved = now
                        message = 'SAVED! Move to a new position or tilt'
                    else:
                        message = 'Already covered: change position, distance or tilt'
                last_corners, last_detection = corners, now
            with lock:
                state.update(current=corners, current_time=stamp, status=message)
        with lock:
            points, names, size = list(state['points']), list(state['names']), state['size']
            state.update(phase='calculating', current=None, status='Calculating calibration...')
        if len(points) < 12:
            raise RuntimeError(f'Only {len(points)} views; need at least 12. Photos retained.')
        args = SimpleNamespace(cols=a.cols, rows=a.rows, square_mm=a.square_mm,
                               output=a.output, width=size[0], height=size[1], fps=30,
                               sensor_id=0, sensor_mode=1, images=None, source=a.source, pipeline=None)
        K,D,rms,errors = calibration.calibrate(args, points, names, size)
        first = cv2.imread(names[0])
        cv2.imwrite(str(a.output/'undistorted_preview.png'), cv2.undistort(first,K,D,None,K))
        message = f'Complete: {len(points)} views, RMS {rms:.3f} px'
        if rms > 1 or max(errors)>2:
            message += ' - high error; review needed'
        with lock:
            state.update(phase='complete', status=message, rms=float(rms))
        print(message, flush=True)
    except Exception as exc:
        with lock:
            state.update(phase='error', status=str(exc))
        print('ERROR:', exc, flush=True)


PAGE = b'''<!doctype html><meta name="viewport" content="width=device-width"><title>Camera calibration</title>
<style>body{margin:0;background:#111;color:white;font:16px system-ui;text-align:center}img{width:100%;max-width:1280px}p{margin:12px}</style>
<p>Saved poses: numbered cyan outlines. Current detection: yellow. Move, tilt, then hold still.</p>
<img src="/stream"><p id="status">Connecting...</p>
<script>setInterval(async()=>{try{let s=await(await fetch('/status')).json();document.getElementById('status').textContent=s.count+' / '+s.target+' views | '+s.status}catch(e){}},1000)</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            body, kind = PAGE, 'text/html'
        elif self.path == '/status':
            with lock:
                data={k:state[k] for k in ('count','status','phase','rms','output')}
                data['stale'] = time.monotonic()-state['frame_time'] > 3
            data['target']=a.views
            body, kind = json.dumps(data).encode(), 'application/json'
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            try:
                while True:
                    with lock:
                        frame=state['frame']
                        points=list(state['points'])
                        current=state['current']
                        detected_at=state['current_time']
                        stamp=state['frame_time']
                        message=state['status']
                    if frame is None:
                        time.sleep(.1)
                        continue
                    scale=min(1.,1280/frame.shape[1])
                    image=cv2.resize(frame,None,fx=scale,fy=scale)
                    for i,corners in enumerate(points):
                        xy=(corners.reshape(-1,2)*scale).astype(np.int32)
                        quad=xy[[0,a.cols-1,-1,-a.cols]]
                        cv2.polylines(image,[quad],True,(255,220,0),2,cv2.LINE_AA)
                        cv2.putText(image,str(i+1),tuple(quad[0]),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,220,0),2)
                    if current is not None and time.monotonic()-detected_at < 1:
                        xy=(current.reshape(-1,2)*scale).astype(np.int32)
                        cv2.polylines(image,[xy[[0,a.cols-1,-1,-a.cols]]],True,(0,255,255),3)
                    if time.monotonic()-stamp>3:
                        message='STALE CAMERA - check stream'
                    cv2.rectangle(image,(0,0),(image.shape[1],70),(20,20,20),-1)
                    cv2.putText(image,f'Saved {len(points)}/{a.views} | {a.cols} x {a.rows} corners | {a.square_mm:g} mm',
                                (12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                    cv2.putText(image,message,(12,55),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if ok:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg.tobytes()+b'\r\n')
                    time.sleep(.1)
            except (BrokenPipeError,ConnectionResetError):
                pass
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(body)


server=ThreadingHTTPServer(('0.0.0.0',a.port),Handler)
threading.Thread(target=reader,daemon=True).start()
threading.Thread(target=worker,daemon=True).start()
print(f'Preview on port {a.port}; output {a.output}',flush=True)
server.serve_forever()
