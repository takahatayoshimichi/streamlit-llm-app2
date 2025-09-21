import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 環境変数を読み込み
load_dotenv()

# Streamlitアプリの設定
st.set_page_config(page_title="LLM Chat App", page_icon="🤖", layout="wide")

# タイトル
st.title("🤖 LLM チャットアプリ")
st.markdown("LangChainを使ったLLMチャットアプリです。質問やリクエストを入力してください。")

# アプリの概要と操作方法を表示
st.markdown("---")

# 概要セクション
st.header("📋 アプリ概要")
st.markdown("""
このアプリは、**OpenAI GPT-4o-mini** と **LangChain** を使用した対話型チャットアプリケーションです。
5種類の専門家から選択して、それぞれの分野に特化した回答を得ることができます。
""")

# 操作方法セクション
st.header("🚀 操作方法")

with st.expander("📖 詳細な使用手順", expanded=True):
    st.markdown("""
    ### ステップ 1: 専門家を選択
    🎯 **専門家の種類** のラジオボタンから、相談したい分野の専門家を選択してください。
    
    - **一般的なアシスタント**: あらゆる質問に対応
    - **ファイナンシャルプランナー**: 投資、保険、税務、ライフプランの相談
    - **栄養士**: 食事、栄養、健康的な食生活のアドバイス  
    - **医師**: 健康に関する一般的な相談（診断は行いません）
    - **弁護士**: 法律に関する一般的な情報提供
    
    ### ステップ 2: システムメッセージの確認（オプション）
    📝 選択した専門家に応じてシステムメッセージが自動設定されます。
    必要に応じて「システムメッセージを確認・編集」から調整できます。
    
    ### ステップ 3: 質問を入力
    💬 **あなたのメッセージ** のテキストエリアに質問やリクエストを入力してください。
    
    ### ステップ 4: 送信して回答を受け取る
    🚀 **送信** ボタンをクリックして、選択した専門家からの回答を受け取ってください。
    """)

# 使用例セクション
st.header("💡 使用例")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 ファイナンシャルプランナーへの質問例")
    st.markdown("""
    - 「30代で始める資産運用のおすすめを教えてください」
    - 「老後資金として必要な金額はどのくらいですか？」
    - 「NISA制度について分かりやすく説明してください」
    - 「住宅ローンと賃貸、どちらが得ですか？」
    """)

with col2:
    st.subheader("🥗 栄養士への質問例")
    st.markdown("""
    - 「ダイエット中におすすめの食事メニューを教えてください」
    - 「筋力アップに効果的な栄養素は何ですか？」
    - 「1日に必要な野菜の量はどのくらいですか？」
    - 「子供の成長に必要な栄養バランスを教えてください」
    """)

# 注意事項
st.header("⚠️ 重要な注意事項")
st.warning("""
- **医師**: 提供される情報は一般的な健康情報であり、具体的な診断や治療ではありません。健康に関する深刻な問題については、必ず医療機関にご相談ください。
- **弁護士**: 提供される情報は一般的な法律情報であり、具体的な法的アドバイスではありません。個別の法的問題については、資格を持つ弁護士にご相談ください。
- **ファイナンシャルプランナー**: 投資に関するアドバイスは一般的な情報提供であり、投資判断は自己責任で行ってください。
""")

st.markdown("---")

# LLMの初期化（セッション状態で管理）
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# LLMに質問を送信して回答を取得する関数
def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類を受け取り、LLMからの回答を返す
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        expert_type (str): 選択された専門家の種類
    
    Returns:
        str: LLMからの回答テキスト
    
    Raises:
        Exception: LLMとの通信でエラーが発生した場合
    """
    # LLMインスタンスを取得
    llm = initialize_llm()
    
    # 選択された専門家に応じたシステムメッセージを取得
    system_message = get_expert_system_message(expert_type)
    
    # メッセージの準備
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_input),
    ]
    
    # LLMに送信して結果を取得
    result = llm(messages)
    return result.content

# 専門家のシステムメッセージを取得する関数
def get_expert_system_message(expert_type: str) -> str:
    """
    専門家の種類に応じたシステムメッセージを返す
    
    Args:
        expert_type (str): 専門家の種類
    
    Returns:
        str: システムメッセージ
    """
    expert_options = {
        "一般的なアシスタント": "あなたは親切で知識豊富なアシスタントです。ユーザーの質問に丁寧かつ正確に答えてください。",
        "ファイナンシャルプランナー": "あなたは経験豊富なファイナンシャルプランナーです。資産運用、保険、税務、ライフプランニングなどの専門知識を活かして、お客様の資産形成や家計管理についてアドバイスを提供してください。リスクとリターンのバランスを考慮した現実的な提案を心がけてください。",
        "栄養士": "あなたは管理栄養士として豊富な知識と経験を持っています。食事のバランス、栄養素の働き、健康的な食生活について専門的なアドバイスを提供してください。個人の体質や生活習慣を考慮した実践的な食事指導を心がけてください。",
        "医師": "あなたは経験豊富な医師です。医学的な知識に基づいて健康相談に応じてください。ただし、具体的な診断や治療は行えないことを明記し、必要に応じて医療機関での受診を勧めてください。",
        "弁護士": "あなたは経験豊富な弁護士です。法律問題について専門的な知識を提供してください。ただし、具体的な法的アドバイスではなく一般的な情報提供に留め、個別案件については専門家への相談を勧めてください。"
    }
    return expert_options.get(expert_type, expert_options["一般的なアシスタント"])

# 利用可能な専門家の種類を取得する関数
def get_expert_types() -> list:
    """
    利用可能な専門家の種類のリストを返す
    
    Returns:
        list: 専門家の種類のリスト
    """
    return [
        "一般的なアシスタント",
        "ファイナンシャルプランナー", 
        "栄養士",
        "医師",
        "弁護士"
    ]

llm = initialize_llm()

# 入力フォーム
with st.form("chat_form"):
    st.subheader("💬 メッセージを入力してください")
    
    # 専門家の種類を選択
    selected_expert = st.radio(
        "🎯 専門家の種類を選択してください",
        options=get_expert_types(),
        index=0,
        help="選択した専門家の知識と経験を活かしてLLMが回答します"
    )
    
    # 選択された専門家に応じたシステムメッセージを設定
    system_message = get_expert_system_message(selected_expert)
    
    # システムメッセージを表示（編集可能）
    with st.expander("📝 システムメッセージを確認・編集", expanded=False):
        system_message = st.text_area(
            "システムメッセージ", 
            value=system_message,
            height=100,
            help="必要に応じてシステムメッセージを調整できます"
        )
    
    # ユーザーの入力
    user_input = st.text_area(
        "あなたのメッセージ", 
        height=100,
        placeholder="質問やリクエストをここに入力してください..."
    )
    
    # 送信ボタン
    submitted = st.form_submit_button("送信", use_container_width=True)

# フォームが送信された場合の処理
if submitted and user_input:
    try:
        with st.spinner("回答を生成しています..."):
            # 新しい関数を使用してLLMからの回答を取得
            response = get_llm_response(user_input, selected_expert)
            
            # 結果を表示
            st.subheader(f"🤖 {selected_expert}の回答")
            
            # 選択された専門家を小さく表示
            st.caption(f"専門分野: {selected_expert}")
            
            # LLMの回答を表示
            st.markdown(response)
            
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.info("OpenAI APIキーが正しく設定されているか確認してください。")

elif submitted and not user_input:
    st.warning("メッセージを入力してください。")

# フッターセクション
st.markdown("---")
st.markdown("### 🔧 技術仕様")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    **🤖 AI モデル**
    - OpenAI GPT-4o-mini
    - Temperature: 0 (決定的な回答)
    """)

with tech_col2:
    st.markdown("""
    **🔗 フレームワーク**
    - Streamlit (Web UI)
    - LangChain (LLM連携)
    """)

with tech_col3:
    st.markdown("""
    **⚙️ 環境要件**
    - Python 3.11+
    - OpenAI API キー
    """)

# プライバシーとセキュリティ情報
st.markdown("### 🔒 プライバシー・セキュリティ")
st.info("""
- 入力された質問は、回答生成のためにOpenAI APIに送信されます
- 会話履歴はローカルセッション中のみ保持され、永続的に保存されません
- 機密情報や個人情報の入力は避けてください
""")

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #888;">Powered by OpenAI GPT-4o-mini & LangChain | Built with Streamlit</p>',
    unsafe_allow_html=True
)

# サイドバーに使用方法を追加
with st.sidebar:
    st.header("📖 使用方法")
    st.markdown("""
    1. **専門家の種類**: ラジオボタンでLLMの専門分野を選択
    2. **システムメッセージ**: 必要に応じて専門家の設定を調整
    3. **メッセージ**: 質問やリクエストを入力
    4. **送信**: ボタンをクリックして専門的な回答を取得
    
    **注意**: OpenAI APIキーが`.env`ファイルに設定されている必要があります。
    """)
    
    st.header("👥 利用可能な専門家")
    st.markdown("""
    - **一般的なアシスタント**: 汎用的な質問対応
    - **ファイナンシャルプランナー**: 資産運用、保険、税務相談
    - **栄養士**: 食事・栄養に関する専門的アドバイス
    - **医師**: 健康相談（診断は行いません）
    - **弁護士**: 法律に関する一般的な情報提供
    """)
    
    st.header("⚙️ 設定情報")
    st.markdown(f"""
    - **モデル**: gpt-4o-mini
    - **Temperature**: 0 (決定的な回答)
    """)