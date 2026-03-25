/* ============================================
   多模态文档问答助手 - 前端逻辑
   ============================================ */

const API_BASE = '';  // 同源部署，留空

// --- DOM 元素 ---
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const btnSend = document.getElementById('btn-send');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const btnImage = document.getElementById('btn-image');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const clearImageBtn = document.getElementById('clear-image');
const fileList = document.getElementById('file-list');
const statusIndicator = document.getElementById('status-indicator');
const uploadModal = document.getElementById('upload-modal');
const progressFill = document.getElementById('progress-fill');
const uploadStatusText = document.getElementById('upload-status-text');
const btnCloseModal = document.getElementById('btn-close-modal');

let pendingImageBase64 = null;
let isProcessing = false;

// --- 工具函数 ---
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setStatus(text, className) {
    statusIndicator.textContent = `● ${text}`;
    statusIndicator.className = `status-${className}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- 消息渲染 ---
function appendMessage(role, content, sources = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let html = '';
    if (role === 'user') {
        html += `<div class="message-content">${escapeHtml(content)}</div>`;
    } else {
        html += `<div class="message-content">${escapeHtml(content)}`;
        if (sources && sources.length > 0) {
            html += buildSourcesHtml(sources);
        }
        html += `</div>`;
    }
    msgDiv.innerHTML = html;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function appendTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai';
    msgDiv.id = 'typing-indicator';
    msgDiv.innerHTML = `<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function buildSourcesHtml(sources) {
    let html = '<details class="sources"><summary>📎 引用来源 (' + sources.length + ')</summary><ul>';
    sources.forEach((s, i) => {
        html += `<li><strong>[${i + 1}]</strong> ${escapeHtml(s.source || '未知')} <span style="color:#999">score: ${(s.score || 0).toFixed(3)}</span>`;
        html += `<div class="source-chunk">${escapeHtml(s.content || '')}</div></li>`;
    });
    html += '</ul></details>';
    return html;
}

// --- 知识库问答 ---
async function sendChat(question) {
    if (isProcessing || !question.trim()) return;
    isProcessing = true;
    btnSend.disabled = true;
    setStatus('思考中', 'processing');

    appendMessage('user', question);
    appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const data = await res.json();

        removeTypingIndicator();

        if (res.ok) {
            appendMessage('ai', data.answer, data.sources);
        } else {
            appendMessage('ai', `❌ 请求失败：${data.detail || '未知错误'}`);
            setStatus('错误', 'error');
            return;
        }
        setStatus('就绪', 'ready');
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnSend.disabled = false;
    }
}

// --- 图片即时问答 ---
async function sendImageChat(question, imageBase64) {
    if (isProcessing) return;
    isProcessing = true;
    btnSend.disabled = true;
    setStatus('视觉理解中', 'processing');

    // 显示用户消息（含图片）
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user';
    msgDiv.innerHTML = `<div class="message-content">${escapeHtml(question || '请描述这张图片')}<img class="message-image" src="${imageBase64}" alt="用户图片"></div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();

    appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/chat/image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question || '请描述这张图片的内容',
                image_base64: imageBase64.split(',')[1]  // 去掉 data:image/xxx;base64, 前缀
            })
        });
        const data = await res.json();
        removeTypingIndicator();

        if (res.ok) {
            appendMessage('ai', data.answer);
        } else {
            appendMessage('ai', `❌ 请求失败：${data.detail || '未知错误'}`);
            setStatus('错误', 'error');
            return;
        }
        setStatus('就绪', 'ready');
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnSend.disabled = false;
    }
}

// --- 文件上传 ---
async function uploadFiles(files) {
    if (isProcessing || files.length === 0) return;
    isProcessing = true;

    uploadModal.classList.remove('hidden');
    btnCloseModal.classList.add('hidden');
    progressFill.style.width = '0%';

    const total = files.length;
    let successCount = 0;

    for (let i = 0; i < total; i++) {
        const file = files[i];
        uploadStatusText.textContent = `正在处理：${file.name} (${i + 1}/${total})`;
        progressFill.style.width = `${((i) / total) * 100}%`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/ingest/file`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                successCount++;
                addFileToList(file.name, true);
            } else {
                addFileToList(file.name, false, data.detail);
            }
        } catch (err) {
            addFileToList(file.name, false, err.message);
        }
    }

    progressFill.style.width = '100%';
    uploadStatusText.textContent = `处理完成：${successCount}/${total} 成功`;
    btnCloseModal.classList.remove('hidden');
    setStatus('就绪', 'ready');
    isProcessing = false;
}

function addFileToList(name, success, error = null) {
    // 清除空提示
    const hint = fileList.querySelector('.empty-hint');
    if (hint) hint.remove();

    const div = document.createElement('div');
    div.className = 'file-item';

    const ext = name.split('.').pop().toLowerCase();
    let icon = '📄';
    if (['pdf'].includes(ext)) icon = '📕';
    if (['png', 'jpg', 'jpeg'].includes(ext)) icon = '🖼️';
    if (['txt', 'md'].includes(ext)) icon = '📝';

    div.innerHTML = `
        <span class="file-icon">${icon}</span>
        <span class="file-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        <span class="file-status ${success ? '' : 'error'}">${success ? '✓' : '✕'}</span>
    `;

    if (error) {
        div.title = error;
    }

    fileList.appendChild(div);
}

// --- 图片预览管理 ---
function handleImageSelect(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        pendingImageBase64 = e.target.result;
        previewImg.src = pendingImageBase64;
        imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// --- 事件绑定 ---
btnSend.addEventListener('click', () => {
    const text = messageInput.value.trim();
    if (pendingImageBase64) {
        sendImageChat(text, pendingImageBase64);
        clearImagePreview();
    } else if (text) {
        sendChat(text);
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        btnSend.click();
    }
});

// 自动增高
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadFiles(e.target.files);
        e.target.value = '';
    }
});

btnImage.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageSelect(e.target.files[0]);
        e.target.value = '';
    }
});

function clearImagePreview() {
    pendingImageBase64 = null;
    previewImg.src = '';
    imagePreview.classList.add('hidden');
}

clearImageBtn.addEventListener('click', clearImagePreview);

btnCloseModal.addEventListener('click', () => {
    uploadModal.classList.add('hidden');
});

// --- 初始化：加载已有文件列表 ---
async function loadFileList() {
    try {
        const res = await fetch(`${API_BASE}/ingest/files`);
        if (res.ok) {
            const data = await res.json();
            if (data.files && data.files.length > 0) {
                fileList.querySelector('.empty-hint')?.remove();
                data.files.forEach(f => addFileToList(f.name, true));
            }
        }
    } catch (e) {
        // 静默失败，不影响使用
    }
}

loadFileList();
