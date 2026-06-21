const cameraInput = document.querySelector("#cameraInput");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");
const intervalInput = document.querySelector("#intervalInput");
const ocrModeInput = document.querySelector("#ocrModeInput");
const cudaInput = document.querySelector("#cudaInput");
const cameraButton = document.querySelector("#cameraButton");
const detectButton = document.querySelector("#detectButton");
const captureButton = document.querySelector("#captureButton");
const uploadInput = document.querySelector("#uploadInput");
const uploadButton = document.querySelector("#uploadButton");
const addDataButton = document.querySelector("#addDataButton");
const vehicleInfoPanel = document.querySelector("#vehicleInfoPanel");
const vehicleOwnerText = document.querySelector("#vehicleOwnerText");
const vehicleTaxText = document.querySelector("#vehicleTaxText");
const newVehicleForm = document.querySelector("#newVehicleForm");
const plateNumberInput = document.querySelector("#plateNumberInput");
const ownerNameInput = document.querySelector("#ownerNameInput");
const plateDateInput = document.querySelector("#plateDateInput");
const plateText = document.querySelector("#plateText");
const dateText = document.querySelector("#dateText");
const fpsText = document.querySelector("#fpsText");
const statusText = document.querySelector("#statusText");
const videoStream = document.querySelector("#videoStream");
const stillPanel = document.querySelector("#stillPanel");
const stillImage = document.querySelector("#stillImage");
const totalDetections = document.querySelector("#totalDetections");
const todayDetections = document.querySelector("#todayDetections");
const latestPlate = document.querySelector("#latestPlate");

let currentState = {};
let configTimer = null;

function payload() {
  return {
    camera: Number(cameraInput.value || 0),
    threshold: Number(thresholdInput.value || 0.55),
    interval: Number(intervalInput.value || 700),
    ocrMode: ocrModeInput.value || "accurate",
    cuda: cudaInput.checked,
  };
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function scheduleConfig() {
  clearTimeout(configTimer);
  configTimer = setTimeout(async () => {
    try {
      const state = await postJson("/api/config", payload());
      renderState(state);
    } catch (error) {
      statusText.textContent = `Konfigurasi gagal: ${error.message}`;
    }
  }, 180);
}

function renderState(state) {
  currentState = state;
  const busy = Boolean(state.loadingModel || state.processingImage);
  cameraButton.textContent = state.running ? "Stop Kamera" : "Mulai Kamera";
  detectButton.textContent = state.detecting || state.loadingModel ? "Stop Deteksi" : "Mulai Deteksi";
  detectButton.disabled = !state.running || state.processingImage;
  captureButton.disabled = !state.running || busy;
  uploadButton.disabled = busy;
  addDataButton.disabled = busy || !state.hasPendingDetection;
  cameraInput.disabled = state.running || busy;
  ocrModeInput.disabled = busy;
  cudaInput.disabled = Boolean(state.detectorLoaded || state.running || busy);

  plateText.textContent = state.plate || "-";
  dateText.textContent = state.date || "-";
  fpsText.textContent = Number(state.fps || 0).toFixed(1);
  statusText.textContent = state.status || "-";
  renderVehicleState(state);

  thresholdValue.textContent = Number(state.threshold || thresholdInput.value).toFixed(2);
  if (state.camera !== undefined && cameraInput.value !== String(state.camera)) {
    cameraInput.value = String(state.camera);
  }
  if (state.ocrMode && ocrModeInput.value !== state.ocrMode) {
    ocrModeInput.value = state.ocrMode;
  }

  if (state.lastImageUrl) {
    stillImage.src = `${state.lastImageUrl}?t=${Date.now()}`;
    stillPanel.hidden = false;
  }
}

function renderVehicleState(state) {
  const vehicle = state.vehicle;
  const status = state.registrationStatus;

  if (vehicle) {
    vehicleInfoPanel.hidden = false;
    newVehicleForm.hidden = true;
    vehicleOwnerText.textContent = vehicle.owner_name || "-";
    const taxLabel = vehicle.tax_status?.label || "Tidak Diketahui";
    vehicleTaxText.textContent = `Pajak: ${taxLabel} | Masa pajak: ${vehicle.plate_date || "-"}`;
    addDataButton.textContent = "Data Sudah Terdaftar";
    addDataButton.disabled = true;
    return;
  }

  vehicleInfoPanel.hidden = true;
  vehicleOwnerText.textContent = "-";
  vehicleTaxText.textContent = "-";

  if (status === "unregistered" && state.hasPendingDetection) {
    newVehicleForm.hidden = false;
    addDataButton.textContent = "Tambah Data Kendaraan";
    if (!plateNumberInput.value || plateNumberInput.dataset.fromDetection !== state.plate) {
      plateNumberInput.value = state.plate && state.plate !== "-" ? state.plate : "";
      plateNumberInput.dataset.fromDetection = state.plate || "";
    }
    if (!plateDateInput.value || plateDateInput.dataset.fromDetection !== state.date) {
      plateDateInput.value = state.date && state.date !== "-" ? state.date : "";
      plateDateInput.dataset.fromDetection = state.date || "";
    }
    return;
  }

  newVehicleForm.hidden = true;
  addDataButton.textContent = "Tambah Data Kendaraan";
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    renderState(await response.json());
  } catch (error) {
    statusText.textContent = `Server tidak merespons: ${error.message}`;
  }
}

function renderHistory(data) {
  totalDetections.textContent = data.stats.total;
  todayDetections.textContent = data.stats.today;
  latestPlate.textContent = data.stats.latest || "-";
}

async function refreshHistory() {
  try {
    const response = await fetch("/api/history");
    if (response.ok) {
      renderHistory(await response.json());
    }
  } catch (_error) {
  }
}

cameraButton.addEventListener("click", async () => {
  cameraButton.disabled = true;
  try {
    const state = currentState.running
      ? await postJson("/api/camera/stop")
      : await postJson("/api/camera/start", payload());

    renderState(state);
    if (state.running) {
      videoStream.src = `/video?t=${Date.now()}`;
    }
  } catch (error) {
    statusText.textContent = `Kamera error: ${error.message}`;
  } finally {
    cameraButton.disabled = false;
  }
});

detectButton.addEventListener("click", async () => {
  detectButton.disabled = true;
  try {
    const state = currentState.detecting || currentState.loadingModel
      ? await postJson("/api/detection/stop")
      : await postJson("/api/detection/start", payload());
    if (state.ok === false) {
      statusText.textContent = state.message || "Deteksi belum bisa dimulai";
    } else {
      renderState(state);
    }
  } catch (error) {
    statusText.textContent = `Deteksi error: ${error.message}`;
  } finally {
    detectButton.disabled = !currentState.running;
  }
});

captureButton.addEventListener("click", async () => {
  captureButton.disabled = true;
  statusText.textContent = "Mengambil gambar dari kamera...";
  try {
    const state = await postJson("/api/capture", payload());
    if (state.ok === false) {
      statusText.textContent = state.message || "Capture gagal";
    } else {
      renderState(state);
    }
  } catch (error) {
    statusText.textContent = `Capture error: ${error.message}`;
  } finally {
    captureButton.disabled = !currentState.running || currentState.loadingModel || currentState.processingImage;
  }
});

uploadButton.addEventListener("click", async () => {
  const file = uploadInput.files[0];
  if (!file) {
    statusText.textContent = "Pilih file gambar dulu";
    return;
  }

  uploadButton.disabled = true;
  statusText.textContent = "Mengupload dan memproses gambar...";

  const formData = new FormData();
  formData.append("image", file);
  formData.append("threshold", payload().threshold);
  formData.append("interval", payload().interval);
  formData.append("ocrMode", payload().ocrMode);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const state = await response.json();
    if (!response.ok) {
      throw new Error(state.message || "Upload gagal");
    }
    renderState(state);
  } catch (error) {
    statusText.textContent = `Upload error: ${error.message}`;
  } finally {
    uploadButton.disabled = currentState.loadingModel || currentState.processingImage;
  }
});

addDataButton.addEventListener("click", async () => {
  if (!plateNumberInput.value.trim()) {
    statusText.textContent = "Nomor plat wajib diisi";
    plateNumberInput.focus();
    return;
  }

  if (!ownerNameInput.value.trim()) {
    statusText.textContent = "Nama pemilik wajib diisi";
    ownerNameInput.focus();
    return;
  }

  addDataButton.disabled = true;
  statusText.textContent = "Menambahkan data ke database...";
  try {
    const state = await postJson("/api/detections/add", {
      plateNumber: plateNumberInput.value.trim(),
      ownerName: ownerNameInput.value.trim(),
      plateDate: plateDateInput.value.trim(),
    });
    if (state.ok === false) {
      statusText.textContent = state.message || "Data gagal ditambahkan";
    } else {
      plateNumberInput.value = "";
      ownerNameInput.value = "";
      renderState(state);
      await refreshHistory();
    }
  } catch (error) {
    statusText.textContent = `Tambah data error: ${error.message}`;
  } finally {
    addDataButton.disabled = !currentState.hasPendingDetection || currentState.loadingModel || currentState.processingImage;
  }
});

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  scheduleConfig();
});

intervalInput.addEventListener("change", scheduleConfig);
cameraInput.addEventListener("change", scheduleConfig);
ocrModeInput.addEventListener("change", scheduleConfig);
cudaInput.addEventListener("change", scheduleConfig);

refreshStatus();
refreshHistory();
setInterval(refreshStatus, 700);
