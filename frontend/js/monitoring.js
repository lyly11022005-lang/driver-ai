
const camera = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const overlayContext = overlay.getContext("2d");

const placeholder = document.getElementById("cameraPlaceholder");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const testAlertBtn = document.getElementById("testAlertBtn");

const driverStatus = document.getElementById("driverStatus");
const detectionInfo = document.getElementById("detectionInfo");

const alertOverlay = document.getElementById("alertOverlay");
const alertMessage = document.getElementById("alertMessage");
const alarmAudio = document.getElementById("alarmAudio");

let cameraStream = null;
let socket = null;
let sendTimer = null;
let alertTimer = null;
let sendingFrame = false;
let aiReady = false;
let realAlert = false;

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

// ======================================
// HÀM HỖ TRỢ ĐỊNH DẠNG DỮ LIỆU
// ======================================

function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || value === "") {
        return "--";
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? number.toFixed(digits)
        : "--";
}

function getYawnPercent(value) {
    if (value === null || value === undefined || value === "") {
        return "--";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) return "--";

    return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

// ======================================
// 1. VẼ KHUNG NHẬN DIỆN
// ======================================

function drawBox(box, color, label, value, position = "top") {
    if (!box) return;

    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = 3;

    overlayContext.strokeRect(
        box.x,
        box.y,
        box.width,
        box.height
    );

    const text = `${label}: ${value}`;

    overlayContext.font = "bold 14px Arial";

    const textWidth = overlayContext.measureText(text).width;
    const labelWidth = textWidth + 12;
    const labelHeight = 23;

    let labelX = box.x;
    let labelY = box.y - labelHeight;

    // Đặt nhãn mắt trái về phía trái khung mắt
    if (position === "left") {
        labelX = box.x - labelWidth - 6;
        labelY = box.y;

        if (labelX < 0) {
            labelX = box.x;
            labelY = box.y - labelHeight;
        }
    }

    // Đặt nhãn mắt phải về phía phải khung mắt
    if (position === "right") {
        labelX = box.x + box.width + 6;
        labelY = box.y;

        if (labelX + labelWidth > overlay.width) {
            labelX = box.x;
            labelY = box.y - labelHeight;
        }
    }

    // Giữ nhãn nằm trong canvas
    labelX = Math.max(
        0,
        Math.min(labelX, overlay.width - labelWidth)
    );

    labelY = Math.max(
        0,
        Math.min(labelY, overlay.height - labelHeight)
    );

    overlayContext.fillStyle = color;

    overlayContext.fillRect(
        labelX,
        labelY,
        labelWidth,
        labelHeight
    );

    overlayContext.fillStyle = "#FFFFFF";

    overlayContext.fillText(
        text,
        labelX + 6,
        labelY + 16
    );
}

// ======================================
// 2. TÍNH KHOẢNG CÁCH HAI MẮT
// ======================================

function calculateEyeDistance(leftBox, rightBox) {
    if (!leftBox || !rightBox) return null;

    const leftCenterX = leftBox.x + leftBox.width / 2;
    const leftCenterY = leftBox.y + leftBox.height / 2;

    const rightCenterX = rightBox.x + rightBox.width / 2;
    const rightCenterY = rightBox.y + rightBox.height / 2;

    const dx = rightCenterX - leftCenterX;
    const dy = rightCenterY - leftCenterY;

    return Math.sqrt(dx * dx + dy * dy);
}

// ======================================
// 3. CẬP NHẬT BẢNG BÊN PHẢI
// ======================================

function updateRightPanel(data, eyeDistance) {
    const eyeStatusValue = document.getElementById("eyeStatusValue");
    const eyeValue = document.getElementById("eyeValue");

    const mouthStatusValue = document.getElementById("mouthStatusValue");
    const mouthValue = document.getElementById("mouthValue");

    const alertCountValue = document.getElementById("alertCountValue");
    const yawnValue = document.getElementById("yawnValue");
    const eyeDistanceValue = document.getElementById("eyeDistanceValue");

    const faceFound = Boolean(data.face_found);

    if (eyeStatusValue) {
        eyeStatusValue.textContent = !faceFound
            ? "CHƯA XÁC ĐỊNH"
            : (data.eyes_closed ? "NHẮM MẮT" : "MẮT MỞ");
    }

    if (eyeValue) {
        eyeValue.textContent = faceFound
            ? `EAR: ${formatNumber(data.ear)}`
            : "EAR: --";
    }

    if (mouthStatusValue) {
        mouthStatusValue.textContent = !faceFound
            ? "CHƯA XÁC ĐỊNH"
            : (data.yawn_detected ? "ĐANG NGÁP" : "BÌNH THƯỜNG");
    }

    if (mouthValue) {
        mouthValue.textContent = faceFound
            ? `MAR: ${formatNumber(data.mar)}`
            : "MAR: --";
    }

    // Lấy số cảnh báo trực tiếp từ AI server
    if (alertCountValue) {
        alertCountValue.textContent =
            data.alert_count ?? "--";
    }

    if (yawnValue) {
        yawnValue.textContent = faceFound
            ? getYawnPercent(data.yawn_probability)
            : "--";
    }

    if (eyeDistanceValue) {
        eyeDistanceValue.textContent =
            faceFound && eyeDistance !== null
                ? `${eyeDistance.toFixed(1)} px`
                : "-- px";
    }
}

// ======================================
// 4. BẢNG THÔNG SỐ AI DƯỚI CAMERA
// ======================================

function updateStatsPanel(data) {
    let panel = document.getElementById("aiStatsPanel");

    if (!panel) {
        panel = document.createElement("div");
        panel.id = "aiStatsPanel";

        const cameraContainer = camera.closest(".camera-container");

        if (cameraContainer) {
            cameraContainer.insertAdjacentElement("afterend", panel);
        }
    }

    const faceFound = Boolean(data.face_found);

    const ear = faceFound ? formatNumber(data.ear) : "--";
    const mar = faceFound ? formatNumber(data.mar) : "--";
    const probability = faceFound
        ? getYawnPercent(data.yawn_probability)
        : "--";

    panel.innerHTML = `
        <div style="
            margin-top: 12px;
            padding: 16px;
            border-radius: 12px;
            background: #17212D;
            color: white;
            font-family: Arial, sans-serif;
        ">
            <h3 style="margin: 0 0 14px;">
                THÔNG SỐ PHÂN TÍCH AI
            </h3>

            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 10px;
            ">
                <div style="background:#263747;padding:12px;border-radius:8px;">
                    <div style="color:#CBD5E1;font-size:13px;">EAR - MẮT</div>
                    <strong style="font-size:22px;color:${data.eyes_closed ? '#F87171' : '#4ADE80'}">
                        ${ear}
                    </strong>
                    <div style="font-size:12px;">Ngưỡng: ≤ 0.20</div>
                </div>

                <div style="background:#263747;padding:12px;border-radius:8px;">
                    <div style="color:#CBD5E1;font-size:13px;">MAR - MIỆNG</div>
                    <strong style="font-size:22px;color:${data.yawn_detected ? '#F87171' : '#4ADE80'}">
                        ${mar}
                    </strong>
                    <div style="font-size:12px;">Ngưỡng: ≥ 0.30</div>
                </div>

                <div style="background:#263747;padding:12px;border-radius:8px;">
                    <div style="color:#CBD5E1;font-size:13px;">XÁC SUẤT NGÁP</div>
                    <strong style="font-size:22px;color:#FBBF24">
                        ${probability}
                    </strong>
                    <div style="font-size:12px;">Ngưỡng CNN: 50%</div>
                </div>

                <div style="background:#263747;padding:12px;border-radius:8px;">
                    <div style="color:#CBD5E1;font-size:13px;">TRẠNG THÁI</div>
                    <strong style="font-size:17px;color:${data.is_alert ? '#F87171' : '#4ADE80'}">
                        ${faceFound
                            ? (data.is_alert ? 'CẢNH BÁO' : 'BÌNH THƯỜNG')
                            : 'CHƯA XÁC ĐỊNH'}
                    </strong>
                    <div style="font-size:12px;">
                        ${data.alert_message || 'Chưa phát hiện dấu hiệu nguy hiểm'}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ======================================
// 5. VẼ KHUNG MẮT, MIỆNG VÀ KHOẢNG CÁCH
// ======================================

function drawDetectionBoxes(data) {
    if (!overlay || !overlayContext) return;

    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    if (!data.face_found) {
        updateRightPanel(data, null);
        updateStatsPanel(data);
        return;
    }

    const scaleX = overlay.width / data.frame_width;
    const scaleY = overlay.height / data.frame_height;

    function scaleBox(box) {
        if (!box) return null;

        return {
            x: box.x * scaleX,
            y: box.y * scaleY,
            width: box.width * scaleX,
            height: box.height * scaleY
        };
    }

    const leftEye = scaleBox(data.left_eye_box);
    const rightEye = scaleBox(data.right_eye_box);
    const mouth = scaleBox(data.mouth_box);

    const earText = formatNumber(data.ear);
    const marText = formatNumber(data.mar);
    const yawnText = getYawnPercent(data.yawn_probability);

    const eyeColor = data.eyes_closed
        ? "#EF4444"
        : "#22C55E";

    const mouthColor = data.yawn_detected
        ? "#EF4444"
        : "#22C55E";

    // Mắt trái: nhãn ưu tiên nằm phía trái
    drawBox(
        leftEye,
        eyeColor,
        "Mắt trái",
        `EAR ${earText}`,
        "left"
    );

    // Mắt phải: nhãn ưu tiên nằm phía phải
    drawBox(
        rightEye,
        eyeColor,
        "Mắt phải",
        `EAR ${earText}`,
        "right"
    );

    // Vẽ đường nối tâm hai mắt
    const eyeDistance = calculateEyeDistance(
        leftEye,
        rightEye
    );

    if (leftEye && rightEye && eyeDistance !== null) {
        const leftX = leftEye.x + leftEye.width / 2;
        const leftY = leftEye.y + leftEye.height / 2;

        const rightX = rightEye.x + rightEye.width / 2;
        const rightY = rightEye.y + rightEye.height / 2;

        overlayContext.save();

        overlayContext.beginPath();
        overlayContext.moveTo(leftX, leftY);
        overlayContext.lineTo(rightX, rightY);

        overlayContext.strokeStyle = "#38BDF8";
        overlayContext.lineWidth = 2;
        overlayContext.setLineDash([5, 4]);
        overlayContext.stroke();

        overlayContext.setLineDash([]);

        overlayContext.fillStyle = "#38BDF8";
        overlayContext.font = "bold 13px Arial";

        const centerX = (leftX + rightX) / 2;
        const centerY = (leftY + rightY) / 2;

        overlayContext.fillText(
            `${eyeDistance.toFixed(1)} px`,
            centerX + 5,
            centerY - 8
        );

        overlayContext.restore();
    }

    // Khung miệng
    drawBox(
        mouth,
        mouthColor,
        "Miệng",
        `MAR ${marText} | Ngáp ${yawnText}`,
        "top"
    );

    updateRightPanel(data, eyeDistance);
    updateStatsPanel(data);
}

// ======================================
// 6. KẾT NỐI AI SERVER
// ======================================

function connectAI() {
    const protocol =
        window.location.protocol === "https:" ? "wss:" : "ws:";

    socket = new WebSocket(
        `${protocol}//${window.location.host}/ws`
    );

    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
        console.log("Đã kết nối WebSocket.");
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "ready") {
            aiReady = true;

            driverStatus.textContent = "AI ĐÃ SẴN SÀNG";

            detectionInfo.textContent =
                "Đang bắt đầu phân tích hình ảnh...";
        }

        if (data.type === "analysis") {
            updateDetection(data);
        }

        if (data.type === "error") {
            console.error("AI error:", data.message);

            detectionInfo.textContent =
                "Lỗi xử lý AI: " + data.message;
        }
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);

        detectionInfo.textContent =
            "Không thể kết nối hệ thống AI.";
    };

    socket.onclose = () => {
        aiReady = false;
        console.log("Đã ngắt kết nối AI.");
    };
}

// ======================================
// 7. GỬI ẢNH CAMERA ĐẾN AI
// ======================================

function sendFrame() {
    if (
        !cameraStream ||
        !aiReady ||
        !socket ||
        socket.readyState !== WebSocket.OPEN ||
        sendingFrame ||
        camera.videoWidth === 0
    ) {
        return;
    }

    sendingFrame = true;

    const originalWidth = camera.videoWidth;
    const originalHeight = camera.videoHeight;

    const scale = Math.min(1, 480 / originalWidth);

    canvas.width = Math.round(originalWidth * scale);
    canvas.height = Math.round(originalHeight * scale);

    context.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob((blob) => {
        if (blob && socket && socket.readyState === WebSocket.OPEN) {
            socket.send(blob);
        }

        sendingFrame = false;
    }, "image/jpeg", 0.75);
}

// ======================================
// 8. CẬP NHẬT KẾT QUẢ NHẬN DIỆN
// ======================================

function updateDetection(data) {
    drawDetectionBoxes(data);

    const isAlert = Boolean(data.is_alert);

    realAlert = isAlert;

    if (isAlert) {
        clearTimeout(alertTimer);

        setAlert(true, data.alert_message);

        driverStatus.textContent = "PHÁT HIỆN BUỒN NGỦ";
    } else {
        setAlert(false);

        if (!data.face_found) {
            driverStatus.textContent = "KHÔNG THẤY KHUÔN MẶT";
        } else {
            driverStatus.textContent = "ĐANG GIÁM SÁT";
        }
    }

    const yawnPercent = getYawnPercent(data.yawn_probability);

    detectionInfo.textContent =
        `${data.eye_status ?? "Chưa xác định"} | ` +
        `${data.mouth_status ?? "Chưa xác định"} | ` +
        `EAR: ${formatNumber(data.ear)} | ` +
        `MAR: ${formatNumber(data.mar)} | ` +
        `Ngáp CNN: ${yawnPercent} | ` +
        `Số cảnh báo: ${data.alert_count ?? "--"}`;
}

// ======================================
// 9. BẬT CAMERA
// ======================================

async function startCamera() {
    if (cameraStream) return;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        camera.srcObject = cameraStream;

        camera.addEventListener("loadedmetadata", () => {
            overlay.width = camera.videoWidth;
            overlay.height = camera.videoHeight;
        }, { once: true });

        await camera.play();

        placeholder.classList.add("hidden");

        driverStatus.textContent = "ĐANG KẾT NỐI AI";

        detectionInfo.textContent =
            "Đang kết nối hệ thống nhận diện...";

        connectAI();

        sendTimer = setInterval(sendFrame, 300);

    } catch (error) {
        console.error(error);

        alert(
            "Không thể bật camera hoặc kết nối AI. " +
            "Vui lòng kiểm tra quyền camera và máy chủ."
        );

        stopCamera();
    }
}

// ======================================
// 10. TẮT CAMERA
// ======================================

function stopCamera() {
    if (sendTimer) {
        clearInterval(sendTimer);
        sendTimer = null;
    }

    if (socket) {
        socket.close();
        socket = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(
            track => track.stop()
        );

        cameraStream = null;
    }

    camera.srcObject = null;

    placeholder.classList.remove("hidden");

    aiReady = false;
    sendingFrame = false;
    realAlert = false;

    setAlert(false);

    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    const statsPanel = document.getElementById("aiStatsPanel");

    if (statsPanel) {
        statsPanel.remove();
    }

    driverStatus.textContent = "CHƯA GIÁM SÁT";

    detectionInfo.textContent =
        "Vui lòng bật camera để bắt đầu giám sát.";

    updateRightPanel({
        face_found: false,
        alert_count: 0
    }, null);
}

// ======================================
// 11. CẢNH BÁO HÌNH ẢNH + ÂM THANH
// ======================================

function setAlert(active, message = "CẢNH BÁO BUỒN NGỦ!") {
    if (active) {
        alertOverlay.classList.remove("hidden");

        alertMessage.textContent = message;

        if (alarmAudio.paused) {
            alarmAudio.loop = true;

            alarmAudio.play().catch(error => {
                console.warn("Không thể phát âm thanh:", error);
            });
        }

    } else {
        alertOverlay.classList.add("hidden");

        alarmAudio.pause();
        alarmAudio.currentTime = 0;
        alarmAudio.loop = false;
    }
}

// ======================================
// 12. NÚT ĐIỀU KHIỂN
// ======================================

startBtn.addEventListener("click", startCamera);

stopBtn.addEventListener("click", stopCamera);

testAlertBtn.addEventListener("click", () => {
    setAlert(true, "ĐANG THỬ CẢNH BÁO!");

    clearTimeout(alertTimer);

    alertTimer = setTimeout(() => {
        if (!realAlert) {
            setAlert(false);
        }
    }, 4000);
});

// Cho phép các phần khác gọi cảnh báo
window.setDriverAIAlert = setAlert;
