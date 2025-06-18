function uploadFile() {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    const resultDiv = document.getElementById('result');
    const uploadBtn = document.querySelector('.upload-btn');

    if (!file) {
        alert('请先选择一个图像文件！');
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.textContent = '上传中...';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '上传并识别';
        return response.json();
    })
    .then(data => {
        if (data.message) {
            alert(data.message);
        } else {
            resultDiv.innerHTML = `
                <p>识别结果：${data.disease}</p>
            `;
        }
    })
    .catch(error => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '上传并识别';
        console.error('Error:', error);
        alert('上传失败，请检查网络连接或文件格式！');
    });
}