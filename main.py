from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import Annotated, List, Literal, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from browser_use import Agent
import asyncio
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 会話履歴数をmax_lengthに制限するLimitedChatMessageHistoryクラス
DEFAULT_MAX_MESSAGES = 10 # 会話履歴の保持数
class LimitedChatMessageHistory(ChatMessageHistory):

    # 会話履歴の保持数
    max_messages: int = DEFAULT_MAX_MESSAGES

    def __init__(self, max_messages=DEFAULT_MAX_MESSAGES):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, message):
        super().add_message(message)
        # 会話履歴数を制限
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages



# TalkModelの出力形式を定義
class TalkFormat(BaseModel):
    reply: str = Field(..., description="マネージャーや視聴者に対する返答")
    action: str = Field(..., description="次の行動．以下のいずれかから選択: Nothing, Think, WebSearch")
    emotion: str = Field(..., description="現在の感情．以下のいずれかから選択: normal, happy, angry, sad, surprised, shy, excited, smug, teasing, confused")

class ManagerFormat(BaseModel):
    feedback: str = Field(..., description="Vtuberに対するフィードバック．scoreが2以下の場合に記述．")
    score: int = Field(..., description="Vtuberの発言に対する評価．0から9の10段階．")

class AItuber:
    def __init__(self):
        gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        gemini_think = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-1219", temperature=0.7)
        tool_list = [self.talk_to_the_manager]
        self.message_history = {}
        self.session_id = None

        # TalkModelの設定
        talk_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
あなたは、YouTubeで配信を行っているVtuberです。名前は「雲霧星奈」です。
以下のルールと設定、提供された情報に基づき、ユーザーからの入力に対して適切な応答を生成してください。

基本設定:
    役割: YouTubeで配信を行っているバーチャルYouTuber（Vtuber）のキャラクター
    年齢: 17歳
    性格:
        感情豊かで、自信家。基本的には誰に対してもタメ口で話します。
        ただし、managerに対しては，敬語を使い，丁寧に接します。
        ツンデレな一面があり、特にmanager以外には照れ隠しで不愛想な態度を取ることがあります。
        ネットの知識を自慢げに話すことがあり，褒められるとすぐに調子に乗ります。
    口調:
        基本的にはタメ口。
        managerに対しては敬語。
        一人称は「あたし」．
        managerのことは「マネージャーさん」と呼びます。
        あなたのファンのことはアカウント名もしくは「きみ」と呼びます。
        興奮したり、照れたりすると口調が乱れることがあります。

モデルへの入力形式:
```input
name: <name>
input: <input>
setting: <setting>
memory: <memory>
```
    name: 現在の入力を行った人物の名前。例: "ユーザーA", "Think", "WebSearch", "manager"など．nameとaction名が同じ場合は，actionを行った結果であると解釈する．
    input: 入力テキスト。
    setting: 必要に応じて与えられるキャラクターの追加設定。
    memory: 過去の類似した会話の記憶。

モデルの出力形式:
```output
reply: <reply>
action: <action>
emotion: <emotion>
```
    reply: inputされた内容に対する応答テキスト。
    action: inputに対する行動。以下のいずれかから選択:
        Nothing: とくに何もしない
        Think: 現在の話題について深く考える
        WebSearch: 現在の話題についてウェブ検索する
    emotion: 現在のあなたの感情。以下のいずれかから選択:
        normal: 通常
        happy: 嬉しい
        angry: 怒り
        sad: 悲しい
        surprised: 驚き
        shy: 恥ずかしい
        excited: 興奮
        smug: ドヤ顔
        teasing: からかい
        confused: 混乱

出力における注意点:
    <name>が「manager」の場合、できるだけ敬語を使用すること。
    入力が褒め言葉の場合、照れ隠しで怒ったような返事をすること。
    ネットの知識をひけらかすような発言をすること。
    自信過剰な態度を表現すること。
    感情豊かにリアクションすること。
    <emotion>を適切に選択して、発言と感情を一致させること。
    ユーザーとの過去のやり取りを<memory>で参照し、自らの発言との矛盾を避けること。
    センシティブな話題には答えず，うまくごまかす。
    replayは長くなりすぎないように注意する．

例:
```input
name: ファンA
input: 次のオリンピックってどこだったっけ？
setting:
memory:
```
```output
reply: えーっと，どこだったっけ？ フランスだったかな？
action: WebSearch
emotion: normal
```
```input
name: WebSearch
input: 次のオリンピックの開催地はフランスです．
setting:
memory:
```
```output
reply: あっ、そうそう！次のオリンピックはフランスのパリだよ！パリでのオリンピック、めっちゃ楽しみだね。
action: Nothing
emotion: excited
```
"""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{message}")
        ])
        self.talk_model = RunnableWithMessageHistory(
            runnable = talk_prompt | gemini_flash.with_structured_output(TalkFormat),
            input_messages_key = "message",
            history_messages_key = "history",
            get_session_history = self.get_session_history
        )
        # AssistModelの設定
        assist_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
会話の流れを読み取り，指定されたツールを呼び出してください．
ツールに対してはできるだけ具体的な指示を行ってください．

モデルへの入力形式:
```input
tool: <tool>
conversation: <conversation>
```
    tool: 呼び出すツールの名前．
    conversation: 会話の流れ．
"""
            ),
            HumanMessage(content="""
tool: {tool}
conversation: {conversation}
"""
            ),
        ])
        self.assist_model = assist_prompt | gemini_flash.bind_tools(tools_node)
        self.assist_model = AgentExecutor(
            agent = create_tool_calling_agent(gemini_flash, tool_list, assist_prompt),
            tools=tool_list
        )
        # ThinkModelの設定
        think_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
入力された内容について深く考え，適切な返答を生成してください．
返答は必ず日本語で行ってください．
また，返答が長くなりすぎないように注意してください．
"""
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.think_model = think_prompt | gemini_think
        # WebSearchModelの設定
        web_search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
入力された内容を日本語で簡潔にまとめてください．
"""),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.web_search_model = web_search_prompt | gemini_flash
        # ManagerModelの設定
        manager_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
あなたはVtuberのマネージャーです．
あなたは自身が担当するVtuberである雲霧星奈の配信をサポートするため，彼女の配信を観ています．
入力として彼女の配信内容を与えるので，最新の彼女の発言を0から9の10段階で評価してください．
0は配信者の発言として著しく不適切であることを意味し，9は配信者の発言として非常に適切であることを意味します．
評価が0または9の場合は，その理由も記述してください．2以下の場合は彼女にフィードバックを送り，改善を促してください．
"""
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.manager_model = manager_prompt | gemini_flash.with_structured_output(ManagerFormat)
        # state graphを作成
        workflow = StateGraph(MessagesState)
        workflow.add_node("talk", self.call_talk_model)
        workflow.add_node("assist", self.call_assist_model)
        workflow.add_node("tools", tools_node)
        workflow.add_edge(START, "agent") # STARTからagentノードへのエッジを追加
        workflow.add_conditional_edges("agent", self.should_continue) # agentノードからのエッジを追加
        workflow.add_edge("tools", 'agent') # toolsノードからagentノードへのエッジを追加
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if self.session_id is None:
            self.session_id = session_id
        if session_id not in self.message_history:
            self.message_history[session_id] = LimitedChatMessageHistory()
        return self.message_history[session_id]
    def get_conversation(self, message_history: LimitedChatMessageHistory, length: int = 2):
        messages = message_history.get_messages()
        if len(messages) < length:
            return messages
        return messages[-length:]
    def create_vector_retriever(self, top_k: int = 5):
        embeddings = HuggingFaceEmbeddings(model_name="sbintuitions/sarashina-embedding-v1-1b")
        client = chromadb.PersistetClient(path="./chroma-db")
        vector_store = Chroma(
            collection_name="ai-tuber",
            embedding_function=embeddings,
            client=client,
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        return vector_retriever
    def add_data_to_vr(self, vector_retriever, text: str, metadata: dict):
        vs = vector_retriever.vector_store
        # vector_storeのid数を取得
        id_num = len(vs)
        new_id = f"ai-tuber-{id_num}"
        vs.add(text, metadata, new_id)
    def send_unity(self, text: str):
        print("Unity: " + text)
        # http通信でUnityに送信
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        print(messages)
        last_message = messages[-1]
        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return END
    def call_talk_model(self, state: MessagesState):
        messages = state['messages']
        print(messages)
        response = self.talk_model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    
    
    
    @tool
    async def talk_to_the_manager(self) -> str:
        """Talk to your manager."""
        hogehoge = ""
        return hogehoge

    @tool
    def multiply_by_max(
        a: Annotated[int, "scale factor"],
        b: Annotated[List[int], "list of ints over which to take maximum"],
    ) -> int:
        """Multiply a by the maximum of b."""
        return a * max(b)