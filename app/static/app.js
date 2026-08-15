const state = { jobId: null, result: null, display: "en", poller: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const stages = { queued:"待機中", validating:"ファイル確認中", downloading:"音声取得中", converting:"音声変換中", transcribing:"文字起こし中", diarizing:"話者解析中", translation_queued:"翻訳待機中", translating:"翻訳中", cached:"キャッシュを利用", completed:"完了", cancelling:"キャンセル中", cancelled:"キャンセル済み", failed:"失敗", translation_failed:"翻訳失敗", interrupted:"中断" };

async function api(path, options={}) {
  const response = await fetch(path, options);
  let body; try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(body.detail || "通信に失敗しました。");
  return body;
}
function showError(message) { $("#error").textContent=message; $("#error").classList.remove("hidden"); }
function startProgress(jobId) { state.jobId=jobId; $("#progress-card").classList.remove("hidden"); $("#result-card").classList.add("hidden"); $("#error").classList.add("hidden"); clearInterval(state.poller); poll(); state.poller=setInterval(poll,1200); }
async function poll() {
  try {
    const job=await api(`/api/jobs/${state.jobId}`); const progress=job.progress||0;
    $("#stage").textContent=stages[job.stage]||job.stage; $("#progress-number").textContent=`${progress}%`; $("#progress-bar").style.width=`${progress}%`;
    if (job.status==="completed") { clearInterval(state.poller); state.result=await api(`/api/jobs/${state.jobId}/result`); renderResult(); }
    if (["failed","cancelled"].includes(job.status)) { clearInterval(state.poller); showError(job.error||"処理を完了できませんでした。"); }
    if (job.stage==="translation_failed") showError(job.error);
  } catch(error) { clearInterval(state.poller); showError(error.message); }
}
function clock(seconds) { const n=Math.max(0,Math.floor(seconds)); return `${String(Math.floor(n/3600)).padStart(2,"0")}:${String(Math.floor(n%3600/60)).padStart(2,"0")}:${String(n%60).padStart(2,"0")}`; }
function visibleText(s) { if(state.display==="ja") return s.translation_ja||"（未翻訳）"; if(state.display==="both") return `${s.original}\n\n${s.translation_ja||"（未翻訳）"}`; return s.original; }
function renderResult() {
  const r=state.result; $("#progress-card").classList.add("hidden"); $("#result-card").classList.remove("hidden");
  $("#result-title").textContent=r.title||"文字起こし結果"; $("#meta").textContent=`言語: ${r.detected_language||"不明"} · 長さ: ${clock(r.duration||0)}`;
  $("#segments").innerHTML=r.segments.map(s=>`<article class="segment"><div><div class="who"></div><div class="time">${clock(s.start)} – ${clock(s.end)}</div></div><div class="body"></div></article>`).join("");
  $$(".segment").forEach((node,i)=>{node.querySelector(".who").textContent=r.segments[i].speaker; const parts=visibleText(r.segments[i]).split("\n\n"); node.querySelector(".body").innerHTML=""; parts.forEach((text,j)=>{const p=document.createElement("div");p.className=j?"translated":"original";p.textContent=text;node.querySelector(".body").appendChild(p);});});
  renderSpeakers();
}
function renderSpeakers() {
  const speakers=[...new Set(state.result.segments.map(s=>s.speaker))];
  $("#speaker-editor").innerHTML=speakers.map((s,i)=>`<label class="speaker-edit"><span>${escapeHtml(s)} →</span><input data-speaker="${escapeHtml(s)}" placeholder="名前を変更"><button data-save="${i}">保存</button></label>`).join("");
  $$("[data-save]").forEach(button=>button.onclick=async()=>{const input=button.parentElement.querySelector("input");if(!input.value.trim())return;state.result=await api(`/api/jobs/${state.jobId}/speakers`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({names:{[input.dataset.speaker]:input.value.trim()}})});renderResult();});
}
function escapeHtml(value) { return value.replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }

$$('.tab').forEach(tab=>tab.onclick=()=>{$$('.tab,.source-panel').forEach(x=>x.classList.remove('active'));tab.classList.add('active');$(`#${tab.dataset.tab}-form`).classList.add('active');});
$("#url-form").onsubmit=async event=>{event.preventDefault();try{const job=await api("/api/transcribe/url",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:$("#space-url").value,mode:$("#mode").value,diarize:$("#diarize").checked})});startProgress(job.job_id);}catch(error){$("#progress-card").classList.remove("hidden");showError(error.message);}};
$("#file-form").onsubmit=async event=>{event.preventDefault();const data=new FormData();data.append("file",$("#audio-file").files[0]);data.append("mode",$("#mode").value);data.append("diarize",$("#diarize").checked);try{const job=await api("/api/transcribe/file",{method:"POST",body:data});startProgress(job.job_id);}catch(error){$("#progress-card").classList.remove("hidden");showError(error.message);}};
$("#cancel").onclick=()=>state.jobId&&api(`/api/jobs/${state.jobId}/cancel`,{method:"POST"});
$("#translate").onclick=async()=>{try{const job=await api("/api/translate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:state.jobId})});startProgress(job.job_id);}catch(error){showError(error.message);}};
$$('[data-display]').forEach(button=>button.onclick=()=>{$$('[data-display]').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.display=button.dataset.display;renderResult();});
$$('[data-format]').forEach(button=>button.onclick=()=>{window.location=`/api/jobs/${state.jobId}/export/${button.dataset.format}?display=${state.display}`;});
$("#copy").onclick=async()=>{await navigator.clipboard.writeText(state.result.segments.map(s=>`${s.speaker} [${clock(s.start)}]\n${visibleText(s)}`).join("\n\n"));$("#copy").textContent="コピー済み";setTimeout(()=>$("#copy").textContent="コピー",1500);};
api("/api/health").then(h=>{$("#health").textContent=`${h.gpu?"GPU":"CPU"} · FFmpeg ${h.ffmpeg?"✓":"×"} · Whisper ${h.whisper?"✓":"×"}`;}).catch(()=>$("#health").textContent="環境確認に失敗");
