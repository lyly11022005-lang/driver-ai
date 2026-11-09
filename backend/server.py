
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from backend.ai_engine import DriverAIEngine


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="DRIVER AI",
    description="Hệ thống phát hiện buồn ngủ của tài xế",
    version="1.0.0"
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "project": "DRIVER AI",
        "message": "AI server is running"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    engine = None

    try:
        # Khởi tạo bộ AI riêng cho trình duyệt này
        engine = await asyncio.to_thread(DriverAIEngine)

        await websocket.send_json({
            "type": "ready",
            "message": "DRIVER AI đã sẵn sàng"
        })

        while True:

            # Nhận ảnh JPEG từ camera trình duyệt
            frame_bytes = await websocket.receive_bytes()

            # Chạy AI ngoài luồng xử lý bất đồng bộ
            result = await asyncio.to_thread(
                engine.process_frame,
                frame_bytes
            )

            # Gửi kết quả phân tích về trình duyệt
            await websocket.send_json({
                "type": "analysis",
                **result
            })

    except WebSocketDisconnect:
        print("Một trình duyệt đã ngắt kết nối.")

    except Exception as error:
        print("Lỗi AI:", error)

        try:
            await websocket.send_json({
                "type": "error",
                "message": str(error)
            })
        except Exception:
            pass

    finally:
        if engine is not None:
            await asyncio.to_thread(engine.close)


# Phục vụ giao diện website
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)
