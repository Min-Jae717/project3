
import streamlit as st
import os
import json
import numpy as np
from supabase import create_client
from sentence_transformers import SentenceTransformer

# 페이지 구성
st.set_page_config(page_title="갤럭시탭 검색", layout="wide")

# 환경변수 또는 secrets 사용
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
except:
    import dotenv
    dotenv.load_dotenv()
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    st.error("Supabase 키가 설정되지 않았습니다.")
    st.stop()

supabase = create_client(supabase_url, supabase_key)

# SentenceTransformer 모델 로드
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedding_model = load_embedding_model()

def generate_embedding(text):
    return embedding_model.encode(text).tolist()

def semantic_search(query_text, limit=10, match_threshold=0.5):
    try:
        query_embedding = np.array(generate_embedding(query_text), dtype=float)

        result = supabase.table('documents').select('id, content, metadata, embedding').execute()
        results = []
        for item in result.data:
            if 'embedding' in item and item['embedding'] is not None:
                try:
                    item_embedding = np.array(item['embedding'], dtype=float)
                    similarity = np.dot(query_embedding, item_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
                    )
                    if similarity > match_threshold:
                        results.append({
                            'id': item['id'],
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'similarity': float(similarity)
                        })
                except:
                    continue
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:limit]
        return results
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {str(e)}")
        raise

st.title("갤럭시탭 시맨틱 검색 (SentenceTransformer 기반)")
st.write("Supabase에 저장된 데이터를 무료 SentenceTransformer 임베딩으로 검색합니다.")

st.sidebar.title("검색 설정")
query = st.text_input("검색어 입력", value="갤럭시탭")
col1, col2 = st.columns(2)
with col1:
    limit = st.slider("검색 결과 수", 1, 50, 10)
with col2:
    threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.3, step=0.01)

if st.button("검색"):
    if query:
        with st.spinner("검색 중..."):
            results = semantic_search(query, limit, threshold)
            if results:
                st.success(f"{len(results)}개의 결과를 찾았습니다.")
                for i, result in enumerate(results):
                    similarity = result['similarity'] * 100
                    metadata = result.get('metadata', {})
                    title = metadata.get('title', '제목 없음')
                    url = metadata.get('url', None)
                    with st.expander(f"{i+1}. {title} (유사도: {similarity:.2f}%)"):
                        st.write(f"**내용:** {result['content'][:300]}...")
                        st.write(f"**블로그:** {metadata.get('bloggername', '')}")
                        st.write(f"**날짜:** {metadata.get('date', '')}")
                        if url:
                            st.markdown(f"[원문 보기]({url})")
            else:
                st.warning("검색 결과가 없습니다.")
    else:
        st.warning("검색어를 입력하세요.")

st.sidebar.title("데이터베이스 상태")
try:
    result = supabase.table('documents').select('id', count='exact').execute()
    st.sidebar.info(f"저장된 문서 수: {result.count}개")
except:
    st.sidebar.warning("문서 수를 불러올 수 없습니다.")
