import socketio
import asyncio
from aiohttp import web

sio = socketio.AsyncServer(
    max_http_buffer_size=1e8,
    cors_allowed_origins= '*',
    async_mode='aiohttp',
    ping_interval=60,  
    ping_timeout=1000000  
)

app = web.Application()
sio.attach(app)

@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Headers'] = 'X-Requested-With'
    return response

app.middlewares.append(cors_middleware)

@sio.event
def connect(sid, environ):
    print("connect ", sid)

@sio.event
async def chat_message(sid, data):
    await sio.emit("my_message", {"data": data})
    print("message ", data)
    
@sio.event
async def frame(sid, data):
    await sio.emit("frame_data", data)
    
@sio.event
async def cheat_notif(sid, data):
    await sio.emit("notify", data)

@sio.event
async def start_ujian(sid, data):
    print("Ujian start")
    await sio.emit("start_stream", data)
    
@sio.event
async def stop_ujian(sid, data):
    print("Stop start")
    await sio.emit("stop_stream", data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)
    
    
async def main():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 3000)
    await site.start()

    while True:
        await asyncio.sleep(3600)  

asyncio.run(main())
