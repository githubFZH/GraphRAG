import os
import asyncio
import time
import pandas as pd
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 导入必要的GraphRAG模块和类
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# 创建FastAPI应用
app = FastAPI()

# 设置常量和配置
INPUT_DIR = "/Users/charlesqin/PycharmProjects/RAGCode/inputs/artifacts"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

# 全局变量，用于存储搜索引擎和问题生成器
search_engine = None
question_generator = None


# 定义API请求的数据模型
class Query(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.0


class QuestionGenRequest(BaseModel):
    question_history: List[str]
    question_count: Optional[int] = 5


async def setup_llm_and_embedder():
    """
    设置语言模型（LLM）和嵌入模型
    """
    api_key = os.environ["GRAPHRAG_API_KEY"]
    llm_model = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-3.5-turbo")
    embedding_model = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")

    # 初始化ChatOpenAI实例
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    # 初始化token编码器
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # 初始化文本嵌入模型
    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    return llm, token_encoder, text_embedder


async def load_context():
    """
    加载上下文数据，包括实体、关系、报告、文本单元和协变量
    """
    # 读取实体数据
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # 设置和加载实体描述嵌入
    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

    # 读取关系数据
    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    # 读取社区报告数据
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    # 读取文本单元数据
    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    # 读取和处理协变量数据
    covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
    claims = read_indexer_covariates(covariate_df)
    print(f"Claim records: {len(claims)}")
    covariates = {"claims": claims}

    return entities, relationships, reports, text_units, description_embedding_store, covariates


async def setup_search_engine(llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                              description_embedding_store, covariates):
    """
    设置搜索引擎，包括上下文构建器和搜索参数
    """
    # 初始化上下文构建器
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    # 设置本地上下文参数
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    # 设置语言模型参数
    llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    # 初始化本地搜索引擎
    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    return search_engine, context_builder, llm_params, local_context_params


@app.on_event("startup")
async def startup_event():
    """
    应用启动时的初始化事件
    """
    global search_engine, question_generator

    # 设置语言模型和嵌入器
    llm, token_encoder, text_embedder = await setup_llm_and_embedder()

    # 加载上下文数据
    entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context()

    # 设置搜索引擎
    search_engine, context_builder, llm_params, local_context_params = await setup_search_engine(
        llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
        description_embedding_store, covariates
    )

    # 设置问题生成器
    question_generator = LocalQuestionGen(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )


@app.post("/v1/completions")
async def create_completion(query: Query):
    """
    处理文本补全请求的API端点
    """
    if not search_engine:
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    try:
        # 执行搜索
        result = await search_engine.asearch(query.prompt)
        # 返回格式化的响应
        return {
            "id": "cmpl-" + os.urandom(12).hex(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": "graphrag-local-search",
            "choices": [
                {
                    "text": result.response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(query.prompt.split()),
                "completion_tokens": len(result.response.split()),
                "total_tokens": len(query.prompt.split()) + len(result.response.split())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/question_generation")
async def generate_questions(request: QuestionGenRequest):
    """
    处理问题生成请求的API端点
    """
    if not question_generator:
        raise HTTPException(status_code=500, detail="问题生成器未初始化")

    try:
        # 生成候选问题
        candidate_questions = await question_generator.agenerate(
            question_history=request.question_history,
            context_data=None,
            question_count=request.question_count
        )
        # 返回格式化的响应
        return {
            "id": "qgen-" + os.urandom(12).hex(),
            "object": "question_generation",
            "created": int(time.time()),
            "model": "graphrag-question-generator",
            "choices": [
                {
                    "questions": candidate_questions.response,
                    "index": 0
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 使用uvicorn运行FastAPI应用
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8012)