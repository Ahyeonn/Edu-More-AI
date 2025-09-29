# client.py (로컬 테스트용)
import asyncio
import websockets

async def main():
    # uri = "ws://127.0.0.1:8000/ws/hand"

    uri = "ws://172.30.1.32:8000/ws/hand"
    async with websockets.connect(uri) as ws:
        # 필요하면 시작 메시지도 보낼 수 있음
        # await ws.send('{"cmd":"start"}')

        while True:
            msg = await ws.recv()
            print(msg)
            if msg == "[END]":
                break

asyncio.run(main())
