
import streamlit as st
import os
import json
import numpy as np
from supabase import create_client
from openai import OpenAI

# 페이지 구성
st.set_page_config(page_title="갤럭시탭 검색", layout="wide")

# Streamlit에서 실행 중인지 확인하고 secrets 가져오기
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    try:
        import dotenv
        dotenv.load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    except:
        st.error("API 키를 가져오는 데 실패했습니다.")
        st.stop()

if not supabase_url or not supabase_key or not openai_api_key:
    st.error("필요한 API 키가 설정되지 않았습니다.")
    st.stop()

try:
    supabase = create_client(supabase_url, supabase_key)
    st.sidebar.success("Supabase 연결 성공!")
except Exception as e:
    st.error(f"Supabase 연결 중 오류: {str(e)}")
    st.stop()

try:
    openai_client = OpenAI(api_key=openai_api_key)
    st.sidebar.success("OpenAI 연결 성공!")
except Exception as e:
    st.error(f"OpenAI 연결 중 오류: {str(e)}")
    st.stop()

def generate_embedding(text):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 생성 중 오류: {str(e)}")
        raise

def semantic_search(query_text, limit=10, match_threshold=0.5):
    try:
        query_embedding = generate_embedding(query_text)
        query_embedding = np.array(query_embedding, dtype=float)

        try:
            response = supabase.rpc(
                'match_documents', 
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_threshold': match_threshold,
                    'match_count': limit
                }
            ).execute()

            if response.data and len(response.data) > 0:
                st.sidebar.success("RPC 검색 성공!")
                return response.data
        except Exception as e:
            st.sidebar.warning(f"RPC 검색 실패, 클라이언트 측에서 검색: {str(e)}")

        st.sidebar.info("문서 가져오는 중...")
        result = supabase.table('documents').select('id, content, metadata, embedding').execute()

        st.sidebar.info(f"{len(result.data)}개 문서에서 유사도 계산 중...")
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
                except Exception as e:
                    continue

        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:limit]
        return results

    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {str(e)}")
        raise

st.title("갤럭시탭 시맨틱 검색")
st.write("Supabase 벡터 데이터베이스에 저장된 갤럭시탭 관련 블로그 데이터를 시맨틱 검색합니다.")

st.sidebar.title("검색 설정")
query = st.text_input("검색어 입력", value="갤럭시탭", help="검색할 키워드나 문장을 입력하세요")

col1, col2 = st.columns(2)
with col1:
    limit = st.slider("검색 결과 수", min_value=1, max_value=50, value=10)
with col2:
    threshold = st.slider("유사도 임계값", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("검색", key="search_button"):
    if query:
        with st.spinner("검색 중..."):
            try:
                results = semantic_search(query, limit=limit, match_threshold=threshold)

                if results:
                    st.success(f"{len(results)}개의 결과를 찾았습니다.")
                    for i, result in enumerate(results):
                        similarity = result['similarity'] * 100
                        metadata = result.get('metadata', {})
                        title = metadata.get('title', '제목 없음')
                        url = metadata.get('url', None)

                        with st.expander(f"{i+1}. {title} (유사도: {similarity:.2f}%)"):
                            st.write(f"**내용:** {result['content'][:300]}...")

                            meta_col1, meta_col2 = st.columns(2)
                            with meta_col1:
                                if 'bloggername' in metadata:
                                    st.write(f"**블로그:** {metadata['bloggername']}")
                                if 'date' in metadata:
                                    st.write(f"**날짜:** {metadata['date']}")
                            with meta_col2:
                                if url:
                                    st.markdown(f"**링크:** [원본 글 보기]({url})")
                                if 'collection' in metadata:
                                    st.write(f"**컬렉션:** {metadata['collection']}")
                else:
                    st.warning("검색 결과가 없습니다.")
            except Exception as e:
                st.error(f"검색 중 오류 발생: {str(e)}")
    else:
        st.warning("검색어를 입력하세요.")

st.sidebar.title("데이터베이스 상태")
try:
    result = supabase.table('documents').select('id', count='exact').execute()
    doc_count = result.count if hasattr(result, 'count') else len(result.data)
    st.sidebar.info(f"저장된 문서 수: {doc_count}개")
except Exception as e:
    st.sidebar.error("데이터베이스 상태를 확인할 수 없습니다.")

st.sidebar.title("사용 안내")
st.sidebar.info("""
1. 검색어 입력: 검색할 키워드나 문장을 입력하세요.
2. 검색 결과 수: 원하는 결과 수를 설정하세요.
3. 유사도 임계값: 높은 값일수록 관련성이 높은 결과만 표시됩니다.
""")
