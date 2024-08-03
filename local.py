import os
import asyncio
import pandas as pd
import tiktoken
from rich import print
from typing import List

# 导入必要的模块和类
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


async def run_search(search_engine, query: str):
    """
    执行搜索查询
    """
    result = await search_engine.asearch(query)
    return result


async def generate_questions(question_generator, history: List[str]):
    """
    基于历史生成新的问题
    """
    questions = await question_generator.agenerate(
        question_history=history, context_data=None, question_count=5
    )
    return questions


async def main():
    """
    主函数，运行整个搜索和问题生成流程
    """
    try:
        # 设置语言模型和嵌入器
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()

        # 加载上下文数据
        entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context()

        # 设置搜索引擎
        search_engine, context_builder, llm_params, local_context_params = await setup_search_engine(
            llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
            description_embedding_store, covariates
        )

        # 运行搜索示例
        queries = [
            "how to take a screenshot of the page in crawl4ai?",
            "how to set a custom user agent in crawl4ai?",
            "tell me what is Extraction Strategies and show some examples in crawl4ai.",
        ]

        for query in queries:
            print(f"\n[bold]Query:[/bold] {query}")
            result = await run_search(search_engine, query)
            print(f"[bold]Response:[/bold]\n{result.response}")
            print("\n[bold]Context Data:[/bold]")
            print("Entities:")
            print(result.context_data["entities"].head())
            print("\nRelationships:")
            print(result.context_data["relationships"].head())
            print("\nReports:")
            print(result.context_data["reports"].head())
            print("\nSources:")
            print(result.context_data["sources"].head())
            if "claims" in result.context_data:
                print("\nClaims:")
                print(result.context_data["claims"].head())

        # 问题生成
        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
        )

        question_history = [
            "how to take a screenshot of the page in crawl4ai?",
            "how to set a custom user agent in crawl4ai?",
            "tell me what is Extraction Strategies and show some examples in crawl4ai.",
        ]
        print("\n[bold]Generating questions based on history:[/bold]")
        print(f"History: {question_history}")
        candidate_questions = await generate_questions(question_generator, question_history)
        print("Generated questions:")
        for i, question in enumerate(candidate_questions.response, 1):
            print(f"{i}. {question}")

    except Exception as e:
        print(f"[bold red]An error occurred:[/bold red] {str(e)}")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())