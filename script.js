const CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
];

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let stream = null;

const button = document.getElementById('start-button');
button.onclick = async function () {
    if (stream) {
        stopCamera();
    } else {
        await startCamera();
    }
};

const SIZE = 320;
const inputCanvas = document.createElement('canvas');
inputCanvas.width = SIZE;
inputCanvas.height = SIZE;
const inputCtx = inputCanvas.getContext('2d', {
    willReadFrequently: true
});

function preprocessFrame(){
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const scale = Math.min(SIZE / vw, SIZE / vh);
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    const padX = Math.floor((SIZE - w) / 2);
    const padY = Math.floor((SIZE - h) / 2);

    inputCtx.fillStyle = 'rgb(114, 114, 114)';
    inputCtx.fillRect(0, 0, SIZE, SIZE);    
    inputCtx.drawImage(video, padX, padY, w, h);

    const pixels = inputCtx.getImageData(
        0, 0, SIZE, SIZE
    ).data;

    const data = new Float32Array(3 * SIZE * SIZE);
    const area = SIZE * SIZE;
    for (let i = 0; i < area; i++) {
        data[i] = pixels[i * 4] / 255;
        data[area + i] = pixels[i * 4 + 1] / 255;
        data[2 * area + i] = pixels[i * 4 + 2] / 255;
}
return {
    tensor: new ort.Tensor(
        'float32', data, [1, 3, SIZE, SIZE]
        ),
        scale,
        padX,
        padY
    };
}

let session = null;
async function loadModel() {
    try {
        status.textContent = 'Loading YOLO Model...';
        session = await ort.InferenceSession.create(
        './yolo11n.onnx',
        {
            executionProviders: ['wasm'],
        });
        console.log('Model Ready');
        console.log('Inputs:', session.inputNames);
        console.log('Outputs:', session.outputNames);

        status.textContent = 'YOLO model ready';
        } catch (error) {
            console.error('Error loading model:', error);
            status.textContent = 'Error loading model: ' + error.message;
        }
}

async function startCamera() {
    button.disabled = true;
    try {
        stream  = await navigator.mediaDevices.getUserMedia({
            audio:false,
            video:{
                facingMode: 'environment',
            }
        });

        video.srcObject = stream;
        await video.play();


        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        status.textContent = 'Camera Running';
        if (session) {
            detecting = true;
            detectionLoop();
        } else {
            status.textContent = 'Camera running, YOLO not loaded';
        }

    }catch(error){
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            video.srcObject = null;
            status.textContent = 'Camera error: ' + error.message;
            button.textContent = 'Start Detection';
        } finally {
            button.disabled = false;
        }
}


function stopCamera() {
    detecting = false;
    if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

    video.pause();
    video.srcObject = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    status.textContent = 'Camera Not Started';
    button.textContent = 'Start Detection';
}

async function detectObjects() {
    if (!session || !stream) {
        return;
    }

    const { tensor, scale, padX, padY } =
        preprocessFrame();

    const results = await session.run({
        [session.inputNames[0]]: tensor
    });

    const output = results[session.outputNames[0]];

    const detections = processOutput(
        output.data,
        scale,
        padX,
        padY
    );

    drawDetections(detections);
}

const CONF_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;

function processOutput(data, scale, padX, padY) {
    const count = 2100;
    const detections = [];

    for (let i = 0; i < count; i++) {
        let score = 0;
        let classId = 0;

        for (let c = 0; c < 80; c++) {
            const confidence = data[(c + 4) * count + i];

            if (confidence > score) {
                score = confidence;
                classId = c;
            }
        }

        if (score < CONF_THRESHOLD) continue;

        const cx = data[i];
        const cy = data[count + i];
        const w = data[2 * count + i];
        const h = data[3 * count + i];

        const x = (cx - w / 2 - padX) / scale;
        const y = (cy - h / 2 - padY) / scale;

        detections.push({
            x,
            y,
            w: w / scale,
            h: h / scale,
            score,
            classId
        });
    }

    detections.sort((a, b) => b.score - a.score);

    return nonMaxSuppression(detections);
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);

    const intersection =
        Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    const union =
        a.w * a.h + b.w * b.h - intersection;

    return union > 0 ? intersection / union : 0;
}

function nonMaxSuppression(detections) {
    const selected = [];

    for (const detection of detections) {
        if (selected.every(
            other => other.classId !== detection.classId ||
                    calculateIoU(other, detection) < IOU_THRESHOLD
        )) {
            selected.push(detection);
        }
    }

    return selected;
}

function getClassColor(classId) {
    const hue = (classId * 137.508) % 360;
    return `hsl(${hue}, 100%, 50%)`;
}
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 3;
    ctx.font = '18px Arial';

    for (const d of detections) {
        const color = getClassColor(d.classId);
        ctx.strokeStyle = color;
        ctx.strokeRect(d.x, d.y, d.w, d.h);

        const label =
        `${CLASS_NAMES[d.classId]}: ${(d.score * 100).toFixed(1)}%`;

        const labelY = Math.max(20, d.y - 5);

        ctx.fillStyle = color;
        ctx.fillText(label, d.x, labelY);
    }
}


let detecting = false;
async function detectionLoop() {
    while (detecting && stream) {
        try {
            await detectObjects();
        } catch (error) {
            console.error('Detection error:', error);
            status.textContent = 'Detection error: ' + error.message;
            detecting = false;
        }

        // Allow the browser to update the interface.
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

// function drawBoundingBoxes() {
//     ctx.clearRect(
//         0, 0, canvas.width, canvas.height
//     );

//     const x = canvas.width * 0.25;
//     const y = canvas.height * 0.25;
//     const w = canvas.width * 0.5;
//     const h = canvas.height * 0.5;

//     ctx.strokeStyle = 'red';
//     ctx.lineWidth = 4;
//     ctx.strokeRect(x, y, w, h);

//     ctx.fillStyle = 'red';
//     ctx.font = '20px Arial';
//     ctx.fillText('Detected Object', x, y > 30 ? y - 10 : y + 30); 
// }

loadModel();