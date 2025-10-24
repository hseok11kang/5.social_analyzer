# 📊 Social Analyzer — (v0.9.6, Streamlit)
# ------------------------------------------------------------
# 설치:
# pip install -U streamlit pandas numpy altair pillow wordcloud matplotlib python-dateutil google-genai
# 실행:
# streamlit run 5.social_analyzer.py
# ------------------------------------------------------------

import os, re, html, time, base64, hashlib
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from dateutil.relativedelta import relativedelta

# wordcloud(없으면 막대그래프 대체)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# (선택) Gemini SDK
try:
    from google import genai
    from google.genai import types as gen_types
except Exception:
    genai = None
    gen_types = None

# ------------------ 공통 스타일 ------------------
st.set_page_config(page_title="Social Analyzer", page_icon="📊", layout="wide")

CARD_CSS = """
<style>
.kpi{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.03);
display:flex;flex-direction:column;justify-content:space-between;min-height:110px;margin-bottom:14px}
.kpi .label{color:#6b7280;font-size:13px;margin-bottom:6px}
.kpi .value{font-weight:800;font-size:26px;line-height:1.1}
.kpi .sub{color:#9ca3af;font-size:12px;margin-top:4px}
.card{border:1px solid #e5e7eb;border-radius:16px;padding:16px;background:#fff;margin-bottom:16px}
.card h3{margin:0 0 12px 0}
.modecard{background:#f5f6f8;border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;margin-bottom:12px}
.modehint{color:#6b7280;font-size:12px;margin-top:4px}
.modenote{color:#6b7280;font-size:12px;margin-top:8px;line-height:1.5}
.insight{margin-top:8px;color:#374151;font-size:14px}
.icon{font-size:20px;margin-right:6px}
.iconbar{display:flex;gap:10px;align-items:center}
.boldborder{border-width:2px !important;border-color:#111827 !important}
.ranktitle{display:flex;align-items:center;gap:8px}
.ranktitle .medal{font-size:22px}
.hl-pink{background:#fde2e4;border-radius:4px;padding:0 3px}
.spr-wrap{margin-bottom:16px}
.spr-platform{display:flex;align-items:center;gap:8px;font-weight:700;color:#111827;margin-bottom:8px}
.spr-platform .pf-ico{font-size:18px}
.spr-card{border:1px solid #e5e7eb;background:#fff;border-radius:14px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.spr-profile{display:flex;align-items:center;gap:10px}
.spr-ava{width:28px;height:28px;border-radius:999px;background:#e5e7eb;display:flex;align-items:center;justify-content:center;font-weight:800;color:#374151}
.spr-author{font-weight:700;color:#111827}
.spr-handle{color:#6b7280;font-size:12px;margin-left:8px}
.spr-posttext{font-size:14px;color:#111827;margin:8px 0 10px;line-height:1.45}
.spr-img{border-radius:10px;overflow:hidden;margin-bottom:8px}
.spr-status{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:12px;color:#111827;align-items:center}
.spr-status .muted{color:#6b7280}
.spr-bottom{display:flex;justify-content:flex-end;margin-top:6px}
.capsent{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid transparent}
.capsent.pos{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}
.capsent.neu{background:#f3f4f6;color:#374151;border-color:#e5e7eb}
.capsent.neg{background:#fef2f2;color:#991b1b;border-color:#fecaca}
.center-note{width:100%;text-align:center;margin-top:-6px;color:#111827;font-weight:700}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

ICON = {"X":"<span class='icon'>𝕏</span>", "FB":"<span class='icon'>📘</span>", "IG":"<span class='icon'>📸</span>"}
PF_NAME = {"X":"X", "FB":"Facebook", "IG":"Instagram"}
PF_ICON_CHAR = {"FB":"📘","IG":"📸","X":"𝕏"}
SENT_COLORS = {"positive":"#16a34a", "neutral":"#9ca3af", "negative":"#ef4444"}

def spacer(h=16): st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)

# ------------------ 공통 유틸 ------------------
def load_api_key():
    key = None
    if hasattr(st, "secrets"): key = st.secrets.get("GEMINI_API_KEY", None)
    if not key: key = os.environ.get("GEMINI_API_KEY")
    return key

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    if genai is None or not api_key: return None
    try: return genai.Client(api_key=api_key)
    except Exception: return None

API_KEY = load_api_key()
gclient = get_gemini_client(API_KEY)

def full_range_label(start: date, end: date):
    return f"{start.year}년 {start.month}월 {start.day}일부터 {end.year}년 {end.month}월 {end.day}일까지"

def call_llm_insight(prompt: str, fallback_line: str = None, model: str = "gemini-2.5-flash"):
    prefix = "🔎 "
    if gclient is None:
        return prefix + (fallback_line if fallback_line else "해당 기간 지표는 전월 대비 안정적입니다.")
    try:
        cfg = gen_types.GenerateContentConfig(thinking_config=gen_types.ThinkingConfig(thinking_budget=0))
        resp = gclient.models.generate_content(model=model, contents=prompt, config=cfg)
        text = getattr(resp, "text", "") or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        text = " ".join((text or "").strip().split())
        parts = text.split(". ")
        if len(parts) > 2: text = ". ".join(parts[:2]).rstrip(".") + "."
        return prefix + text
    except Exception:
        return prefix + (fallback_line if fallback_line else "해당 기간 지표는 전월 대비 안정적입니다.")

def seed_from_params(query: str, start: date, end: date, channels: tuple):
    raw = f"{query}|{start.isoformat()}|{end.isoformat()}|{','.join(channels)}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % (10**8)

def clip_int(x, lo, hi): return max(lo, min(int(x), hi))
def humanize(n: int):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return f"{n}"

def fmt_dt(dt):
    try:
        if hasattr(dt, "to_pydatetime"): dt = dt.to_pydatetime()
    except Exception: pass
    if not isinstance(dt, datetime):
        try: dt = pd.to_datetime(dt).to_pydatetime()
        except Exception: dt = datetime.now()
    s = dt.strftime("%I:%M %p %b %d, %Y")
    if s.startswith("0"): s = s[1:]
    return s

# 이미지 유틸
def find_first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def load_crop_to_ratio(img_path: str, ratio=(16,9)) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    rw, rh = ratio; target = rw/rh
    w, h = im.size; cur = w/h
    if cur > target:
        new_w = int(h*target); left = (w-new_w)//2; im = im.crop((left,0,left+new_w,h))
    else:
        new_h = int(w/target); top = (h-new_h)//2; im = im.crop((0,top,w,top+new_h))
    return im

def image_to_base64(im: Image.Image, format="JPEG"):
    from io import BytesIO
    buf = BytesIO(); im.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def truncate_text(s: str, n=220):
    s = s.strip()
    return (s if len(s) <= n else s[:n-1].rstrip() + "…")

# Sprinklr 카드 렌더러(이미지+텍스트 하나의 카드)
def render_spr_post(platform: str, author: str, handle: str, tstamp: datetime,
                    text_html: str, image_path: str, sentiment: str = "positive",
                    likes: int = 0, comments: int = 0, shares: int = 0):
    PF_ICON_CHAR = {"FB":"📘","IG":"📸","X":"𝕏"}
    pill = "pos" if sentiment=="positive" else ("neu" if sentiment=="neutral" else "neg")
    pf_icon = PF_ICON_CHAR.get(platform, "🔗"); pf_name = {"X":"X", "FB":"Facebook", "IG":"Instagram"}.get(platform, platform)
    initials = (author.replace("@"," ").strip()[:2] or "LG").upper()
    im = load_crop_to_ratio(image_path, (16,9))
    b64 = image_to_base64(im, format="JPEG")
    st.markdown(
        f"""
        <div class="spr-wrap">
          <div class="spr-platform"><span class="pf-ico">{pf_icon}</span>{pf_name}</div>
          <div class="spr-card">
            <div class="spr-profile">
              <div class="spr-ava">{initials}</div>
              <div class="spr-author">{html.escape(author)}</div>
              <div class="spr-handle">{html.escape(handle)} • {fmt_dt(tstamp)}</div>
            </div>
            <div class="spr-posttext">{text_html}</div>
            <div class="spr-img">
              <img src="data:image/jpeg;base64,{b64}" style="width:100%;height:auto;border-radius:10px;display:block;" />
            </div>
            <div class="spr-status">
              <div class="muted">✔ Not Set</div><div class="muted">🗂 1 Queue</div>
              <div>🖼 {likes:,}</div><div class="muted">✉ Not Set</div>
              <div class="muted">🛡 Moderation</div><div class="muted">Not Assigned</div>
            </div>
            <div class="spr-bottom"><span class="capsent {pill}">{sentiment.capitalize()}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True
    )

# ------------------ 스프링클러 쿼리 빌더 ------------------
def build_sprinklr_query(brand: str, phrase: str, near_n: int, include_terms: list, exclude_terms: list,
                         channels: list, langs: list):
    brand = brand.strip()
    phrase_q = f"\"{phrase.strip()}\"" if phrase.strip() else ""
    base = f"({brand} NEAR/{near_n} {phrase_q})" if brand and phrase_q else (brand or phrase_q)
    norm = lambda t: (t if (t.startswith('"') and t.endswith('"')) else (f"\"{t.strip()}\"" if " " in t else t.strip()))
    inc = " OR ".join([norm(t) for t in include_terms if t.strip()])
    exc = " OR ".join([norm(t) for t in exclude_terms if t.strip()])
    inc_part = f" AND ({inc})" if inc else ""
    exc_part = f" NOT ({exc})" if exc else ""
    lang_part = f" AND lang:({','.join(langs)})" if langs else ""
    src_map = {"X":"twitter", "FB":"facebook", "IG":"instagram"}
    srcs = [src_map[c] for c in channels if c in src_map]
    src_part = f" AND source:({ ' OR '.join(srcs) })" if srcs else ""
    return (base + inc_part + exc_part + lang_part + src_part).strip()

# ------------------ 더미 데이터 생성 (Listening) ------------------
def _username(rng: np.random.Generator, ch: str):
    first = ["kim","lee","park","choi","jung","han","yoon","kang","oh","lim","moon","song","ryu","hwang","seo"]
    noun = ["tech","home","life","cool","energy","quiet","review","photo","daily","vibes","blog","note","buzz"]
    if ch in ("X","IG"):
        return "@"+rng.choice(first)+ "_" + rng.choice(noun) + str(rng.integers(10,99))
    else:
        cap = rng.choice(first).capitalize(); cap2 = rng.choice(first).capitalize()
        return f"{cap} {cap2}"

def build_listening_long_caption(row, idx:int=0):
    terms = row.get("co_terms", []) if isinstance(row, dict) else row["co_terms"]
    t = [t for t in terms][:4]
    base_ht = "#LG #AirConditioner #DualInverter #EnergySaving #SmartHome #Cooling"
    if row["channel"] == "IG":
        cap = (
            f"First week with the LG air conditioner and I’m already sleeping better. "
            f"Setup was simple and the {', '.join(t[:2])} make a noticeable difference at night. "
            f"Remote + ThinQ app combo is clutch for lazy evenings. {base_ht} #Home #Interior #CozyVibes"
        )
    elif row["channel"] == "X":
        cap = (
            f"Switched to an LG air conditioner this season — bills trending down, comfort way up. "
            f"{', '.join(t)} are the MVP features so far. If you’re debating an upgrade, this is your sign. "
            f"{base_ht} #Review #SummerPrep"
        )
    else:  # FB
        cap = (
            f"We finally installed an LG air conditioner in the living room and the difference is huge. "
            f"The {', '.join(t[:3])} keep our space quiet and cool even during afternoon sun. "
            f"Family approves 👍 {base_ht} #Family #Everyday"
        )
    return cap

@st.cache_data(show_spinner=False)
def generate_dummy_posts(query: str, start: date, end: date, channels: tuple, base_daily=40):
    rng_seed = seed_from_params(query, start, end, channels)
    rng = np.random.default_rng(rng_seed)
    days = (end - start).days + 1
    records = []

    base_terms = [
        "energy efficiency","inverter","low noise","filter","installation","summer","heat wave",
        "power bill","smart control","wifi","warranty","cooling","air quality","PM2.5",
        "remote","price","review","service","design","odor","dehumidify","fast cooling",
        "LG ThinQ","Dual Inverter","Auto Cleaning","R32"
    ]
    ch_like_mu = {"X":18,"FB":26,"IG":40}
    ch_cmt_mu  = {"X":6,"FB":10,"IG":9}
    ch_shr_mu  = {"X":12,"FB":8,"IG":5}
    sentiments = ["positive","neutral","negative"]; sent_probs=[0.48,0.32,0.20]

    # 요일 가중
    day_weights=[]
    for d in range(days):
        the_day = start + timedelta(days=d)
        day_weights.append(1.0 + (0.35 if the_day.weekday()>=5 else 0.0))
    day_weights = np.array(day_weights); day_weights /= day_weights.mean()

    for d in range(days):
        cur_day = start + timedelta(days=d)
        base = base_daily * day_weights[d]
        if rng.random()<0.10: base *= rng.integers(2,5)
        for ch in channels:
            mu = base * (0.36 if ch=="IG" else (0.34 if ch=="X" else 0.30))
            posts = clip_int(rng.normal(mu, mu*0.35), 0, int(mu*3)+5)
            for i in range(posts):
                likes    = clip_int(rng.normal(ch_like_mu[ch], ch_like_mu[ch]*0.6), 0, 5000)
                comments = clip_int(rng.normal(ch_cmt_mu[ch],  ch_cmt_mu[ch]*0.7),  0, 1200)
                shares   = clip_int(rng.normal(ch_shr_mu[ch],  ch_shr_mu[ch]*0.8),  0, 2000)
                co_cnt   = rng.integers(2,6)
                co_terms = rng.choice(base_terms, size=co_cnt, replace=False).tolist()
                sent     = rng.choice(sentiments, p=sent_probs)
                user     = _username(rng, ch)

                records.append({
                    "ts": datetime(cur_day.year, cur_day.month, cur_day.day, rng.integers(0,24), rng.integers(0,60)),
                    "date": cur_day, "channel": ch,
                    "post_id": f"{ch}-{cur_day.isoformat()}-{i}-{rng_seed%9999}",
                    "likes": likes, "comments": comments, "shares": shares,
                    "engagement": likes+comments+shares, "user": user,
                    "co_terms": co_terms, "sentiment": sent
                })
    return pd.DataFrame(records)

def aggregate_trend(df: pd.DataFrame, freq: str):
    g = df.groupby([pd.Grouper(key="ts", freq=freq)]).size().rename("volume").reset_index()
    g["date_label"] = g["ts"].dt.strftime("%b %d")  # Oct 28
    return g

def aggregate_trend_split(df: pd.DataFrame, freq: str, split_col: str):
    g = df.groupby([pd.Grouper(key="ts", freq=freq), split_col]).size().rename("volume").reset_index()
    g["date_label"] = g["ts"].dt.strftime("%b %d")
    return g

def channel_split(df: pd.DataFrame):
    g = df.groupby("channel").size().rename("volume").reset_index().sort_values("volume", ascending=False)
    g["percent"] = (g["volume"] / g["volume"].sum() * 100).round(1)
    return g

def sentiment_split(df: pd.DataFrame):
    g = df.groupby("sentiment").size().rename("volume").reset_index()
    g["sentiment"] = pd.Categorical(g["sentiment"], ["positive","neutral","negative"], ordered=True)
    g = g.sort_values("sentiment")
    g["percent"] = (g["volume"] / g["volume"].sum() * 100).round(1)
    return g

def co_terms_freq(df: pd.DataFrame):
    from collections import Counter
    c = Counter()
    for xs in df["co_terms"]: c.update(xs)
    return dict(c)

# ------------------ Perf Monitoring 더미 ------------------
SUBS = ["HQ","US","KR","IN","BR","TH","VN","EN","DE","FR","IT","ES"]
SUBS_DISPLAY = lambda s: ("UK" if s=="EN" else s)

@st.cache_data(show_spinner=False)
def generate_perf_month(month_start: date, month_end: date):
    month_seed = int(month_start.strftime("%Y%m"))
    rng = np.random.default_rng(month_seed)
    def one_sub(seed_off=0):
        rng_local = np.random.default_rng(month_seed + seed_off)
        posts = int(rng_local.integers(45, 91))
        avg_eng = rng_local.integers(350, 1200)
        engagements = int(posts * avg_eng)
        likes = int(engagements * rng_local.uniform(0.70, 0.80))
        comments = int(engagements * rng_local.uniform(0.10, 0.16))
        shares = max(0, engagements - likes - comments)
        return dict(volume=posts, engagements=engagements, likes=likes, comments=comments, shares=shares)
    def frame_for_all(offset:int):
        rows=[]
        for i,s in enumerate(SUBS):
            r = one_sub(seed_off=(i+1)*100+offset); r["subsidiary"]=s; rows.append(r)
        return pd.DataFrame(rows)
    sub_cur_df  = frame_for_all(0)
    sub_prev_df = frame_for_all(7)
    sub_yoy_df  = frame_for_all(13)
    def kpi_from(df):
        return {"volume":int(df["volume"].sum()),"engagements":int(df["engagements"].sum()),
                "likes":int(df["likes"].sum()),"comments":int(df["comments"].sum()),"shares":int(df["shares"].sum())}
    return kpi_from(sub_cur_df), kpi_from(sub_prev_df), kpi_from(sub_yoy_df), sub_cur_df, sub_prev_df, sub_yoy_df

def pct_change(cur:int, base:int): return 0.0 if base==0 else round((cur - base) / base * 100, 1)

@st.cache_data(show_spinner=False)
def generate_perf_posts(month_start: date, month_end: date):
    rng = np.random.default_rng(int(month_start.strftime("%Y%m")) + 42)
    rows=[]; channels=["IG","X","FB"]
    for i in range(18):
        ch = rng.choice(channels, p=[0.36,0.34,0.30]); sub=rng.choice(SUBS)
        day = month_start + timedelta(days=int(rng.integers(0, max((month_end-month_start).days+1,1))))
        likes=int(rng.integers(300,9000)); comments=int(rng.integers(40,800)); shares=int(rng.integers(60,1000))
        eng=likes+comments+shares; reach=int(eng*rng.uniform(3.0,7.0))
        rows.append({"date":day,"channel":ch,"subsidiary":sub,"likes":likes,"comments":comments,"shares":shares,"engagements":eng,"reach":reach,
                     "caption":"Energy-saving tips with LG air conditioners — short-form video."})
    return pd.DataFrame(rows)

def build_perf_long_caption(row, idx:int=0):
    sd = SUBS_DISPLAY(row["subsidiary"])
    base_ht = "#LG #AirConditioner #DualInverter #EnergySaving #SmartHome #Cooling"
    if idx==0:
        return (f"{sd} team rolled out a quick how-to on keeping rooms cool without hiking the bill. "
                f"Shot in a cozy living room, we showcased smart scheduling and quiet mode in real life. "
                f"Simple tips, strong saves. {base_ht} #HowTo #Reels #Everyday")
    elif idx==1:
        return (f"Nothing beats a quiet afternoon with cool air flowing. "
                f"We compared typical fan modes vs. Dual Inverter performance and the results were clear. "
                f"Compact, efficient, ready for fall. {base_ht} #Comparison #BeforeAfter #Home")
    else:
        return (f"Weekend reset with cleaner air and faster cooling — that’s the combo we love. "
                f"Walkthrough shows setup + ThinQ remote control steps for first timers. "
                f"Save it for later! {base_ht} #ThinQ #StepByStep #Tips")

# ------------------ 상태 ------------------
for k,v in [("listening_ready",False),("listening_df",None),("mode","1) Social Listening"),("query",'LG NEAR/3 "Air Conditioner" NOT promotion'),("pm_show_more_posts",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ------------------ 상단 타이틀 & 모드 ------------------
st.title("📊 Social Analyzer")
st.markdown("<div class='modecard'>", unsafe_allow_html=True)
st.markdown("<b>모드 선택</b><div class='modehint'>원하는 분석 유형을 선택하세요</div>", unsafe_allow_html=True)
st.session_state.mode = st.radio(
    label="",
    options=["1) Social Listening", "2) Social MKT. Performance Monitoring"],
    index=0 if st.session_state.mode.startswith("1") else 1,
    horizontal=False,
    key="mode_radio"
)
st.markdown(
    """
    <div class='modenote'>
    • <b>Social Listening</b> : 특정 토픽에 대한 소셜 미디어 환경 내 언급 현황 및 추이를 확인합니다.<br/>
    • <b>Social MKT. Performance Monitoring</b> : 자사 (HQ + 지역법인)의 소셜 마케팅 현황 및 성과를 확인합니다.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 1) Social Listening
# =========================================================
if st.session_state.mode.startswith("1"):
    spacer()

    # 입력
    c1,c2,c3 = st.columns([2,2,2])
    with c1: period = st.selectbox("기간 프리셋", ["지난 30일","지난 90일","올해","직접 입력"], index=0)
    with c2: gran = st.radio("집계 단위", ["Daily","Weekly","Monthly"], index=0, horizontal=True)
    with c3: channels = st.multiselect("소셜 채널", ["X","FB","IG"], default=["X","FB","IG"])

    today=date.today()
    if period=="지난 30일": start_date,end_date=today-timedelta(days=29),today
    elif period=="지난 90일": start_date,end_date=today-timedelta(days=89),today
    elif period=="올해": start_date,end_date=date(today.year,1,1),today
    else:
        d1,d2 = st.columns(2)
        with d1: start_date = st.date_input("시작일", value=today-timedelta(days=29))
        with d2: end_date   = st.date_input("종료일", value=today)
        if start_date>end_date: st.error("시작일이 종료일보다 뒤일 수 없습니다."); st.stop()
    st.write(f"기간: **{start_date.isoformat()} ~ {end_date.isoformat()}**")
    spacer(8)

    query = st.text_input("키워드(쿼리)", value=st.session_state.query, help='예) LG NEAR/3 "Air Conditioner" NOT promotion')
    with st.expander("🧩 Sprinklr Query Assistant (Beta) — 쿼리 작성이 어려우세요?"):
        qb1,qb2,qb3 = st.columns([2,2,1])
        with qb1:
            qa_brand = st.text_input("브랜드", value="LG")
            qa_phrase = st.text_input("핵심 구문(문구는 자동 \"\" 처리)", value="Air Conditioner")
            qa_near = st.number_input("NEAR/n", min_value=1, max_value=10, value=3, step=1)
        with qb2:
            qa_inc = st.text_area("포함 키워드(쉼표로 구분)", value="inverter, energy efficiency, quiet")
            qa_exc = st.text_area("제외 키워드(쉼표로 구분)", value="promotion, coupon, giveaway")
        with qb3:
            qa_langs = st.multiselect("언어", ["ko","en","es","fr","de","pt"], default=["ko","en"])
            qa_ch = st.multiselect("채널", ["X","FB","IG"], default=["X","FB","IG"])
        if st.button("제안 쿼리 생성"):
            inc_list=[x.strip() for x in qa_inc.split(",")] if qa_inc else []
            exc_list=[x.strip() for x in qa_exc.split(",")] if qa_exc else []
            suggestion=build_sprinklr_query(qa_brand, qa_phrase, qa_near, inc_list, exc_list, qa_ch, qa_langs)
            st.code(suggestion, language="text")
            ap_col,_=st.columns([1,3])
            with ap_col:
                if st.button("⬅ 현재 쿼리에 적용"):
                    st.session_state.query = suggestion; st.success("현재 쿼리에 적용했습니다.")

    run_btn = st.button("분석하기", type="primary")

    def run_listening():
        with st.status("LLM이 크롤링/분석 중입니다…", expanded=True) as status:
            st.write("1/4 수집 중: 채널별 공개 포스트 샘플링…"); time.sleep(1.0)
            st.write("2/4 전처리: 중복·스팸 필터링…"); time.sleep(1.0)
            st.write("3/4 분석: 트렌드·공동언급·감성…"); time.sleep(1.0)
            st.write("4/4 요약: KPI 및 Top 포스트 산출…"); time.sleep(1.0)
            status.update(label="완료", state="complete")
        if not channels: st.warning("채널을 하나 이상 선택하세요."); return
        df = generate_dummy_posts(query, start_date, end_date, tuple(channels))
        st.session_state.listening_df = df
        st.session_state.listening_ready = True

    if run_btn: run_listening()

    if st.session_state.listening_ready and st.session_state.listening_df is not None:
        df = st.session_state.listening_df
        frange = full_range_label(start_date, end_date)

        # 2) Key Numbers
        st.markdown("### 2) Key Numbers")
        st.markdown(f"<div class='iconbar'>{ICON['X']}{ICON['FB']}{ICON['IG']}</div>", unsafe_allow_html=True)
        total_posts=int(df.shape[0])
        total_eng=int((df["likes"]+df["comments"]+df["shares"]).sum())
        total_likes=int(df["likes"].sum()); total_comments=int(df["comments"].sum()); total_shares=int(df["shares"].sum())

        a,b,c,d,e=st.columns(5)
        with a: st.markdown(f"<div class='kpi'><div class='label'>Volume</div><div class='value'>{humanize(total_posts)}</div><div class='sub'>관련 포스트 수</div></div>", unsafe_allow_html=True)
        with b: st.markdown(f"<div class='kpi'><div class='label'>Engagements</div><div class='value'>{humanize(total_eng)}</div><div class='sub'>Likes + Comments + Shares</div></div>", unsafe_allow_html=True)
        with c: st.markdown(f"<div class='kpi'><div class='label'>Likes</div><div class='value'>{humanize(total_likes)}</div><div class='sub'>&nbsp;</div></div>", unsafe_allow_html=True)
        with d: st.markdown(f"<div class='kpi'><div class='label'>Comments</div><div class='value'>{humanize(total_comments)}</div><div class='sub'>&nbsp;</div></div>", unsafe_allow_html=True)
        with e: st.markdown(f"<div class='kpi'><div class='label'>Shares</div><div class='value'>{humanize(total_shares)}</div><div class='sub'>&nbsp;</div></div>", unsafe_allow_html=True)
        kpi_line = f"{frange}의 관련 포스트는 총 {humanize(total_posts)}개 게시되었으며, {humanize(total_eng)}의 Engagement를 발생시켰습니다."
        st.markdown(call_llm_insight("KPI 한줄 요약: "+kpi_line, fallback_line=kpi_line), unsafe_allow_html=True)
        spacer()

        # 3) Social Post Volume Trend
        st.markdown("### 3) Social Post Volume Trend")
        freq = {"Daily":"D","Weekly":"W","Monthly":"M"}[gran]
        tdf = aggregate_trend(df, freq=freq)

        base_line = alt.Chart(tdf).mark_line(point=True).encode(
            x=alt.X("ts:T", title="기간", axis=alt.Axis(format="%b %d")),
            y=alt.Y("volume:Q", title="포스트 수"),
            tooltip=[alt.Tooltip("date_label:N", title="날짜"), alt.Tooltip("volume:Q", title="Volume")]
        ).properties(height=260)

        peak_row = tdf.loc[tdf["volume"].idxmax()]
        peak_df = pd.DataFrame([peak_row])
        peak_df["ai_reason"] = "Oct 06에 Facebook 및 Instagram을 중심으로 LG Airconditioner 신제품에 대한 Buzz가 확산되며 Post 볼륨이 Peak를 기록했습니다."
        peak_df["more"] = "자세히 보기"
        bulb = alt.Chart(peak_df).mark_text(text="💡", dy=-14, fontSize=18).encode(
            x="ts:T", y="volume:Q",
            tooltip=[alt.Tooltip("date_label:N", title="피크"),
                     alt.Tooltip("ai_reason:N", title="AI 해석"),
                     alt.Tooltip("more:N", title=" ")]
        )
        st.altair_chart(base_line + bulb, use_container_width=True)

        tdf_sorted=tdf.sort_values("ts"); n=len(tdf_sorted)
        if n>=4: q1=tdf_sorted.iloc[0]["date_label"]; q2=tdf_sorted.iloc[int(n*0.4)]["date_label"]
        else: q1=tdf_sorted.iloc[0]["date_label"]; q2=tdf_sorted.iloc[-1]["date_label"]
        top2=tdf_sorted.nlargest(2,"volume"); p1=top2.iloc[0]["date_label"]; p2=top2.iloc[1]["date_label"] if len(top2)>1 else p1
        st.markdown(f"<div class='insight'>🔎 {frange}의 트렌드는 {q1}부터 {q2}까지는 비교적 안정적인 흐름을 보이다가, {p1}와 {p2} 때 게시물 수가 급증하는 경향을 보입니다.</div>", unsafe_allow_html=True)

        with st.expander("Channel별 트렌드 보기"):
            tdc = aggregate_trend_split(df, freq=freq, split_col="channel")
            ch_line = alt.Chart(tdc).mark_line(point=True).encode(
                x=alt.X("ts:T", title="기간", axis=alt.Axis(format="%b %d")),
                y=alt.Y("volume:Q", title="포스트 수"),
                color=alt.Color("channel:N", title="채널"),
                tooltip=["date_label:N","channel:N","volume:Q"]
            ).properties(height=260)
            st.altair_chart(ch_line, use_container_width=True)
            ch_peak = (tdc.sort_values("volume", ascending=False).iloc[0] if not tdc.empty else None)
            ch_text = f"🔎 {ch_peak['date_label']}에 정점을 찍었으며, 특히 {ch_peak['channel']} 채널에서의 포스트 수가 급증하는 경향을 보입니다." if ch_peak is not None else "🔎 채널별 특이 피크가 제한적입니다."
            st.markdown(f"<div class='insight'>{ch_text}</div>", unsafe_allow_html=True)

        with st.expander("Sentiment별 트렌드 보기"):
            tds = aggregate_trend_split(df, freq=freq, split_col="sentiment")
            tds["sentiment"] = pd.Categorical(tds["sentiment"], ["positive","neutral","negative"], ordered=True)
            sent_line = alt.Chart(tds).mark_line(point=True).encode(
                x=alt.X("ts:T", title="기간", axis=alt.Axis(format="%b %d")),
                y=alt.Y("volume:Q", title="포스트 수"),
                color=alt.Color("sentiment:N", title="감성",
                                scale=alt.Scale(domain=list(SENT_COLORS.keys()), range=list(SENT_COLORS.values()))),
                tooltip=["date_label:N","sentiment:N","volume:Q"]
            ).properties(height=260)
            st.altair_chart(sent_line, use_container_width=True)
            if not tds.empty:
                grp = tds.groupby("ts")["volume"].sum().reset_index().sort_values("volume", ascending=False)
                s1=grp.iloc[0]["ts"]; s2=grp.iloc[1]["ts"] if len(grp)>1 else s1
                d1=pd.to_datetime(s1).strftime("%b %d"); d2=pd.to_datetime(s2).strftime("%b %d")
                sent_text=f"🔎 {d1}의 급증은 긍정·중립·부정 모두에서 동반 증가했고, {d2}의 급증은 부정 비중이 상대적으로 높아 원인 확인이 필요합니다."
            else:
                sent_text="🔎 감성별로 뚜렷한 급증 패턴은 제한적입니다."
            st.markdown(f"<div class='insight'>{sent_text}</div>", unsafe_allow_html=True)
        spacer()

        # 4) Wordcloud (70% 크기)
        st.markdown("### 4) 동시 언급 키워드 Wordcloud")
        freqs = co_terms_freq(df)
        if freqs and WORDCLOUD_AVAILABLE:
            top5=set([k for k,_ in sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]])
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return "#ef4444" if word in top5 else "#6b7280"
            wc = WordCloud(width=392, height=137, background_color="white",
                           max_words=600, prefer_horizontal=0.95, relative_scaling=0.9,
                           collocations=False, scale=2, margin=0, min_font_size=8
                           ).generate_from_frequencies(freqs)
            wc.recolor(color_func=color_func)
            fig = plt.figure(figsize=(4.9,1.72))
            plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
            st.pyplot(fig, use_container_width=False)
            st.markdown("<div class='insight'>🔎 기능·효율·소음·설치 관련 키워드가 상위권을 구성합니다.</div>", unsafe_allow_html=True)
        elif freqs:
            top = (pd.DataFrame([{"term":k,"count":v} for k,v in freqs.items()]).sort_values("count", ascending=False).head(30))
            chart = alt.Chart(top).mark_bar().encode(
                x=alt.X("count:Q", title="빈도"), y=alt.Y("term:N", sort="-x", title="키워드"),
                tooltip=["term:N","count:Q"]
            ).properties(height=200, width=450)
            st.altair_chart(chart, use_container_width=False)
            st.markdown("<div class='insight'>🔎 워드클라우드 모듈 미설치로 막대형 차트로 대체했습니다.</div>", unsafe_allow_html=True)
        else:
            st.info("동시 언급 키워드 데이터가 없습니다.")
        spacer()

        # 5) 채널 및 Sentiment별 Post 수 비중
        st.markdown("### 5) 채널 및 Sentiment별 Post 수 비중")
        c1,c2=st.columns(2)
        with c1:
            cdf = channel_split(df)
            donut_ch = alt.Chart(cdf).mark_arc(innerRadius=70).encode(
                theta=alt.Theta("volume:Q"),
                color=alt.Color("channel:N", legend=alt.Legend(title="채널")),
                tooltip=[alt.Tooltip("channel:N", title="채널"), alt.Tooltip("volume:Q", title="Post 수"), alt.Tooltip("percent:Q", title="%")]
            ).properties(width=360, height=280, title="채널별 비중")
            st.altair_chart(donut_ch, use_container_width=False)
            ch_read = " · ".join([f"{r['channel']} {int(r['volume']):,}({r['percent']}%)" for _, r in cdf.iterrows()])
            st.markdown(f"<div class='center-note'>총 Post 수 {humanize(total_posts)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight'>🔎 {ch_read}</div>", unsafe_allow_html=True)
        with c2:
            sdf = sentiment_split(df)
            donut_s = alt.Chart(sdf).mark_arc(innerRadius=70).encode(
                theta=alt.Theta("volume:Q"),
                color=alt.Color("sentiment:N", legend=alt.Legend(title="감성"),
                                scale=alt.Scale(domain=list(SENT_COLORS.keys()), range=list(SENT_COLORS.values()))),
                tooltip=[alt.Tooltip("sentiment:N", title="감성"), alt.Tooltip("volume:Q", title="Post 수"), alt.Tooltip("percent:Q", title="%")]
            ).properties(width=360, height=280, title="Sentiment별 비중")
            st.altair_chart(donut_s, use_container_width=False)
            sent_read = " · ".join([f"{r['sentiment']} {int(r['volume']):,}({r['percent']}%)" for _, r in sdf.iterrows()])
            st.markdown(f"<div class='center-note'>총 Post 수 {humanize(total_posts)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight'>🔎 {sent_read}</div>", unsafe_allow_html=True)
        spacer()

        # 7) 인기 포스트 (Top3)
        st.markdown("### 7) 인기 포스트 (Top3)")
        df["reach"] = (df["likes"]*5 + df["comments"]*3 + df["shares"]*6).astype(int)
        topbar_l, topbar_r = st.columns([6,2])
        with topbar_r:
            sort_opt = st.selectbox("정렬 기준", ["Engagement 순","Like 순","Comment 순","Reach 순(추정)"], index=0)
        key_map={"Engagement 순":"engagement","Like 순":"likes","Comment 순":"comments","Reach 순(추정)":"reach"}
        top3 = df.sort_values(key_map[sort_opt], ascending=False).head(3).copy()
        cols = st.columns(3)
        imgs=[["C:/gemini-test/image/Sample4.png","image/Sample4.png"],
              ["C:/gemini-test/image/sample5.png","image/sample5.png"],
              ["C:/gemini-test/image/sample6.png","image/sample6.png"]]
        def highlight_lg_ac_plain(text: str) -> str:
            text = truncate_text(text, 220)
            safe = html.escape(text)
            safe = re.sub(r'(?i)\bLG\b', r"<span class='hl-pink'>\g<0></span>", safe)
            safe = re.sub(r'(?i)\bair\s*conditioner(s)?\b', r"<span class='hl-pink'>\g<0></span>", safe)
            return safe
        for i,(_,row) in enumerate(top3.iterrows()):
            with cols[i]:
                img_path = find_first_existing(imgs[i]) or imgs[i][-1]
                cap = build_listening_long_caption(row, i)
                cap_html = highlight_lg_ac_plain(cap)
                handle = " " + (row["user"] if row["user"].startswith("@") else "@"+row["user"].replace(" ",""))
                render_spr_post(row["channel"], "LuxeGlow", handle, row["ts"], cap_html, img_path,
                                row["sentiment"], int(row["likes"]), int(row["comments"]), int(row["shares"]))

# =========================================================
# 2) Social MKT. Performance Monitoring
# =========================================================
else:
    spacer()

    # 지난달 디폴트
    today=date.today()
    first_this=date(today.year,today.month,1)
    last_end=first_this - timedelta(days=1)
    last_start=date(last_end.year,last_end.month,1)

    d1,d2=st.columns(2)
    with d1: start_pm = st.date_input("시작일", value=last_start)
    with d2: end_pm   = st.date_input("종료일", value=last_end)
    if start_pm>end_pm: st.error("시작일이 종료일보다 뒤일 수 없습니다."); st.stop()

    cur_kpi, prev_kpi, yoy_kpi, sub_cur_df, sub_prev_df, sub_yoy_df = generate_perf_month(start_pm, end_pm)
    # 섹션 공용 게시물 데이터
    posts_all = generate_perf_posts(start_pm, end_pm)

    # ▼▼▼ 추가: 항상 Top3를 보장하는 헬퍼
    def get_top3_posts_for_sub(posts_df: pd.DataFrame, sub: str, key_col: str) -> pd.DataFrame:
        # 1) 해당 법인 상위 정렬
        sel = posts_df[posts_df["subsidiary"]==sub].sort_values(key_col, ascending=False)
        picked = sel.head(3)
        # 2) 부족하면 전체에서 채워서 3개 맞춤
        if len(picked) < 3:
            need = 3 - len(picked)
            rest = posts_df[~posts_df.index.isin(picked.index)].sort_values(key_col, ascending=False).head(need)
            picked = pd.concat([picked, rest], ignore_index=False)
        return picked.head(3)

    # 1) KPI
    st.markdown("### 1) Key Performance Index")
    a,b,c,d,e=st.columns(5)
    def kpi_box(col,label,cur,prev,yoy):
        delta_mom = pct_change(cur, prev); delta_yoy = pct_change(cur, yoy)
        mom_html=f"<span style='color:{('#16a34a' if delta_mom>=0 else '#ef4444')}'>{'+' if delta_mom>=0 else ''}{delta_mom}% MoM</span>"
        yoy_html=f"<span style='color:{('#16a34a' if delta_yoy>=0 else '#ef4444')}'>{'+' if delta_yoy>=0 else ''}{delta_yoy}% YoY</span>"
        with col: st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{humanize(cur)}</div><div class='sub'>{mom_html} · {yoy_html}</div></div>", unsafe_allow_html=True)
    kpi_box(a,"Posts",cur_kpi["volume"],prev_kpi["volume"],yoy_kpi["volume"])
    kpi_box(b,"Engagements",cur_kpi["engagements"],prev_kpi["engagements"],yoy_kpi["engagements"])
    kpi_box(c,"Likes",cur_kpi["likes"],prev_kpi["likes"],yoy_kpi["likes"])
    kpi_box(d,"Comments",cur_kpi["comments"],prev_kpi["comments"],yoy_kpi["comments"])
    kpi_box(e,"Shares",cur_kpi["shares"],prev_kpi["shares"], yoy_kpi["shares"])
    spacer()

    # 2) Subsidiaries — 카드 + 아래 상세(expander)
    st.markdown("### 2) Subsidiaries Social MKT. Performance")
    metric_label = st.selectbox("정렬 기준", ["Engagements","Posts","Likes","Comments","Shares"], index=0)
    METRIC_MAP = {"Posts":"volume","Engagements":"engagements","Likes":"likes","Comments":"comments","Shares":"shares"}
    mcol = METRIC_MAP[metric_label]

    def rank_dict(df, col):
        r=df[["subsidiary",col]].copy(); r["rank"]=r[col].rank(ascending=False, method="dense").astype(int)
        return r.set_index("subsidiary")["rank"].to_dict()
    ranks_cur  = rank_dict(sub_cur_df,  mcol)
    ranks_prev = rank_dict(sub_prev_df, mcol)
    ranks_yoy  = rank_dict(sub_yoy_df,  mcol)

    def ord_suffix(n:int): return "st" if n%10==1 and n%100!=11 else ("nd" if n%10==2 and n%100!=12 else ("rd" if n%10==3 and n%100!=13 else "th"))
    cur_sorted = sub_cur_df.sort_values(mcol, ascending=False).reset_index(drop=True)
    grid_cols = st.columns(3)
    medals={1:"🥇",2:"🥈",3:"🥉"}; card_idx=0

    # 상세용 정렬 키 (Posts 선택 시 engagements로 대체)
    POSTS_METRIC_MAP = {"Engagements":"engagements","Likes":"likes","Comments":"comments","Shares":"shares","Posts":"engagements"}

    for _, row in cur_sorted.iterrows():
        if row["subsidiary"]=="KR":  # 제외 유지
            continue
        sub=row["subsidiary"]; rank=ranks_cur[sub]
        mom = ranks_prev.get(sub, rank) - rank
        yoy = ranks_yoy.get(sub, rank) - rank
        def dstr(d): return f"▲{abs(d)}위" if d>0 else (f"▼{abs(d)}위" if d<0 else "—")
        mom_color="#16a34a" if mom>0 else ("#ef4444" if mom<0 else "#6b7280")
        yoy_color="#16a34a" if yoy>0 else ("#ef4444" if yoy<0 else "#6b7280")
        border_class = " boldborder" if rank in (1,2,3) else ""
        medal = medals.get(rank,""); title=f"{rank}{ord_suffix(rank)}. {('UK' if sub=='EN' else sub)}"

        with grid_cols[card_idx % 3]:
            st.markdown(
                f"""
                <div class='card{border_class}'>
                  <div class='ranktitle'><div class='medal'>{medal}</div><h3>{title}</h3></div>
                  <div>{metric_label}: <b>{int(row[mcol]):,}</b></div>
                  <div class='rankdelta'><span style="color:{mom_color}">MoM {dstr(mom)}</span> · <span style="color:{yoy_color}">YoY {dstr(yoy)}</span></div>
                </div>
                """, unsafe_allow_html=True
            )

            # ▼ 상세(항상 Top3 보장) — 카드 바로 아래 생성
            with st.expander(f"{SUBS_DISPLAY(sub)} 상세 보기 — Top 3 Posts", expanded=False):
                key = POSTS_METRIC_MAP[metric_label]
                p3 = get_top3_posts_for_sub(posts_all, sub, key)
                cols_det = st.columns(3)
                img_candidates = [
                    ["C:/gemini-test/image/Sample1.jpg","image/Sample1.jpg"],
                    ["C:/gemini-test/image/sample2.jpg","image/sample2.jpg"],
                    ["C:/gemini-test/image/sample3.jpg","image/sample3.jpg"],
                ]
                for i,(_,r) in enumerate(p3.reset_index(drop=True).iloc[:3].iterrows()):
                    with cols_det[i]:
                        img_path = find_first_existing(img_candidates[i]) or img_candidates[i][-1]
                        cap = build_perf_long_caption(r, i)
                        render_spr_post(r["channel"], "LuxeGlow", f" @{SUBS_DISPLAY(r['subsidiary']).lower()}",
                                        r["date"], html.escape(truncate_text(cap, 220)), img_path,
                                        "positive", int(r["likes"]), int(r["comments"]), int(r["shares"]))
        card_idx += 1

    spacer()

    # 3) Best Performance Post
    st.markdown("### 3) Best Performance Post")
    topbar_l, topbar_r = st.columns([6,1])
    with topbar_r:
        post_metric = st.selectbox("정렬", ["Engagements","Likes","Comments","Shares","Reach(추정)"], index=0)
    PM_MAP = {"Engagements":"engagements","Likes":"likes","Comments":"comments","Shares":"shares","Reach(추정)":"reach"}
    posts_df = posts_all.sort_values(PM_MAP[post_metric], ascending=False).reset_index(drop=True)
    top3 = posts_df.head(3)
    cols = st.columns(3)
    monitoring_imgs = [
        ["C:/gemini-test/image/Sample1.jpg","image/Sample1.jpg"],
        ["C:/gemini-test/image/sample2.jpg","image/sample2.jpg"],
        ["C:/gemini-test/image/sample3.jpg","image/sample3.jpg"],
    ]
    for i,(_,row) in enumerate(top3.iterrows()):
        with cols[i]:
            img_path = find_first_existing(monitoring_imgs[i]) or monitoring_imgs[i][-1]
            cap = build_perf_long_caption(row, i)
            render_spr_post(row["channel"], "LuxeGlow", f" @{SUBS_DISPLAY(row['subsidiary']).lower()}",
                            row["date"], html.escape(truncate_text(cap, 220)), img_path,
                            "positive", int(row["likes"]), int(row["comments"]), int(row["shares"]))
