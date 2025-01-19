// Function to show loading on button click
function showLoading(buttonId) {
  var button = document.getElementById(buttonId);
  button.disabled = true; // Disable the button
  button.innerHTML = 'Solving... <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
}

// Function to hide loading on button
function hideLoading(buttonId) {
  var button = document.getElementById(buttonId);
  button.disabled = false; // Enable the button
  button.innerHTML = "Solve";
}

// Function to open the camera
function openCamera() {
  var video = document.getElementById("camera-preview");
  var captureButton = document.getElementById("capture-button");
  var solveButton = document.getElementById("solve-button");
  var recaptureButton = document.getElementById("recapture-button");
  var canvas = document.getElementById("camera-canvas");

  // Reset visibility of elements
  video.style.display = "block";
  captureButton.style.display = "inline-block";
  solveButton.style.display = "none";
  recaptureButton.style.display = "none";
  canvas.style.display = "none";

  // Access the camera
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err) {
      console.error("Error accessing the camera: " + err);
      alert("Tidak dapat mengakses kamera. Pastikan kamera terhubung dan izin akses diberikan.");
    });
}

// Function to capture image from video
function captureImage() {
  var video = document.getElementById("camera-preview");
  var canvas = document.getElementById("camera-canvas");
  var captureButton = document.getElementById("capture-button");
  var solveButton = document.getElementById("solve-button");
  var recaptureButton = document.getElementById("recapture-button");

  // Set canvas dimensions to match video dimensions
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw the current video frame onto the canvas
  var context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Stop the video stream
  var stream = video.srcObject;
  var tracks = stream.getTracks();
  tracks.forEach(function (track) {
    track.stop();
  });

  // Hide the video preview and capture button, show the captured image on canvas and solve button
  video.style.display = "none";
  canvas.style.display = "block";
  captureButton.style.display = "none";
  solveButton.style.display = "inline-block";
  recaptureButton.style.display = "inline-block";
}

// Function to recapture image
function recaptureImage() {
  var video = document.getElementById("camera-preview");
  var canvas = document.getElementById("camera-canvas");
  var captureButton = document.getElementById("capture-button");
  var solveButton = document.getElementById("solve-button");
  var recaptureButton = document.getElementById("recapture-button");

  // Reset visibility of elements
  canvas.style.display = "none";
  video.style.display = "block";
  captureButton.style.display = "inline-block";
  solveButton.style.display = "none";
  recaptureButton.style.display = "none";

  // Re-enable the camera
  openCamera();
}

// Function to solve Sudoku by sending captured image to server
function solveSudoku() {
  var canvas = document.getElementById("camera-canvas");
  var imageData = canvas.toDataURL("image/jpeg");

  showLoading("solve-button");

  fetch("http://127.0.0.1:5000/capture", {
    method: "POST",
    body: JSON.stringify({ image: imageData }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      displayResults(data);
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Terjadi kesalahan saat mengirim gambar ke server.");
    })
    .finally(() => {
      hideLoading("solve-button");
      document.getElementById("solve-button").style.display = "none";
      document.getElementById("recapture-button").style.display = "none";
    });
}

// Function to upload image
function uploadImage(event) {
  event.preventDefault();

  // Get the input file
  var input = document.getElementById("image-input");
  var file = input.files[0];

  // Validate file input
  if (!file) {
    alert("Please select an image file.");
    return;
  }

  if (!file.type.startsWith("image/")) {
    alert("Selected file is not an image. Please choose a valid image file.");
    return;
  }

  // Prepare FormData to send image
  var formData = new FormData();
  formData.append("image", file);

  // Show loading indicator
  showLoading("upload-button");

  // Send the image to the backend
  fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      displayResults(data);
    })
    .catch((error) => {
      console.error("Fetch Error Details:", error);
      alert("Terjadi kesalahan saat mengirim gambar ke server: " + error.message);
    })
    .finally(() => {
      hideLoading("upload-button");
    });
}

// Function to display results
function displayResults(data) {
  console.log("Data received from server:", data); // Logging for debugging
  var resultsSection = document.getElementById("results-section");
  resultsSection.innerHTML = "";

  if (data.status && data.status.code !== 200) {
    alert(data.status.message);
    return;
  }

  var images = data.data.images;
  var solution = data.data.solution;

  if (images && images.length > 0) {
    var imagesContainer = document.createElement("div");
    imagesContainer.className = "images-container";

    images.forEach((imagePath) => {
      var img = document.createElement("img");
      img.src = imagePath;
      img.className = "img-thumbnail";
      img.style.cursor = "pointer";
      img.onclick = function () {
        showModal(img);
      };
      imagesContainer.appendChild(img);
    });

    resultsSection.appendChild(imagesContainer);
  }

  if (solution && solution.length > 0) {
    var sudokuContainer = document.createElement("div");
    sudokuContainer.className = "sudoku-container";

    var sudokuTitle = document.createElement("h3");
    sudokuTitle.className = "display-6 text-body-emphasis text-center";
    sudokuTitle.textContent = "Solved Sudoku";

    var table = document.createElement("table");
    table.className = "table table-bordered mx-auto sudoku-table";

    var tbody = document.createElement("tbody");

    solution.forEach((row) => {
      var tr = document.createElement("tr");

      row.forEach((cell) => {
        var td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    sudokuContainer.appendChild(sudokuTitle);
    sudokuContainer.appendChild(table);
    resultsSection.appendChild(sudokuContainer);
  }
}

// Function to show modal with full-size image
function showModal(image) {
  var modalImage = document.getElementById("modalImage");
  modalImage.src = image.src;
  var myModal = new bootstrap.Modal(document.getElementById("imageModal"));
  myModal.show();
}
