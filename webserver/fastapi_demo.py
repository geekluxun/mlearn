from fastapi import FastAPI, HTTPException
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_debug
from langchain_core.globals import set_llm_cache
from langchain_core.globals import set_verbose
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from pydantic import BaseModel

app = FastAPI()


class RequestDto(BaseModel):
    # 定义请求体中需要的数据模型
    q: str
    description: str = None  # 可选字段


@app.post("/api/demo")
async def create_demo_item(request_body: RequestDto):
    if not request_body.q:
        raise HTTPException(status_code=400, detail="q is required.")

    response = llm.invoke({"question": request_body.q})
    return {"request": request_body.q, "response": response}


llm = None


def init():
    set_verbose(True)
    set_debug(False)
    set_llm_cache(InMemoryCache())
    hf = HuggingFacePipeline.from_model_id(
        model_id="/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50},
    )
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | hf
    return chain


if __name__ == "__main__":
    import uvicorn

    llm = init()
    uvicorn.run(app, host="0.0.0.0", port=8000)
