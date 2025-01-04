from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Annotated, List, Literal, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferWindowMemory


# TalkModelの出力形式を定義
class TalkFormat(BaseModel):
    speak: str = Field(..., description="マネージャーや視聴者に対する発言")
    emotion: str = Field(..., description="現在の感情")

class AItuber:
    def __init__(self):
        talk_prompt = ChatPromptTemplate.from_messages([
            ("system", open("prompts/talk_model.txt", "r").read()),
            ("user", "name: {name}\ninput: {input}\ncharacter setting: {setting}\npersonal memory: {personal}\nglobal memory: {global}")
        ])
        gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        gemini_think = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-1219", temperature=0.7)
        tools_node = ToolNode([self.talk_to_the_manager])
        self.talk_model = create_react_agent(
            model=gemini_flash.with_structured_output(TalkFormat),
            tools=tools_node,
            state_modifier=talk_prompt,
            checkpointer=ConversationBufferWindowMemory(k=5),
        )
        # state graphを作成
        workflow = StateGraph(MessagesState)
        workflow.add_node("tools", tools_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_edge("tools", 'agent')

    def send_unity(self, text: str):
        print("Unity: " + text)
        # http通信でUnityに送信
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return END
    def call_talk_model(self, state: MessagesState):
        messages = state['messages']
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