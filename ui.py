import os
import aiohttp
import chainlit as cl
import logging

# 设置环境变量，防止 Chainlit 加载 .env 文件
os.environ["CHAINLIT_AUTO_LOAD_DOTENV"] = "false"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8012"  # 注意：这里使用了8012端口，请确保与您的API端口一致


class CustomAsyncClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.chat = self.Chat(base_url)

    class Chat:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.completions = self.Completions(base_url)

        class Completions:
            def __init__(self, base_url: str):
                self.base_url = base_url

            async def create(self, messages: list, **kwargs):
                logger.info(f"Sending request to {self.base_url}/v1/completions")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                                f"{self.base_url}/v1/completions",
                                json={"prompt": messages[-1]["content"], **kwargs}
                        ) as response:
                            logger.info(f"Received response with status {response.status}")
                            if response.status == 200:
                                data = await response.json()
                                logger.info(f"Response data: {data}")
                                return data
                            else:
                                error_text = await response.text()
                                logger.error(f"API request failed with status {response.status}: {error_text}")
                                raise Exception(f"API request failed with status {response.status}: {error_text}")
                except Exception as e:
                    logger.error(f"Error in create method: {str(e)}")
                    raise


client = CustomAsyncClient(API_BASE_URL)

settings = {
    "model": "graphrag-local-search",
    "temperature": 0,
}


@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f"Received message: {message.content}")
    try:
        # 发送"处理中"的消息
        processing_message = cl.Message(content="Processing your request...")
        await processing_message.send()

        response = await client.chat.completions.create(
            messages=[
                {
                    "content": "You are a helpful bot based on GraphRAG",
                    "role": "system"
                },
                {
                    "content": message.content,
                    "role": "user"
                }
            ],
            **settings
        )

        response_content = response['choices'][0]['text']
        logger.info(f"Sending response: {response_content}")

        # 创建新消息而不是更新旧消息
        await cl.Message(content=response_content).send()

        # 移除"处理中"的消息
        await processing_message.remove()

    except Exception as e:
        logger.error(f"Error in on_message: {str(e)}")
        error_message = f"An error occurred: {str(e)}"
        # 创建新的错误消息
        await cl.Message(content=error_message).send()
        # 移除"处理中"的消息
        await processing_message.remove()


@cl.on_chat_start
async def on_chat_start():
    logger.info("New chat started")
    await cl.Message(content="Welcome! I'm your GraphRAG assistant. How can I help you today?").send()


if __name__ == "__main__":
    logger.info("Starting Chainlit application")
    cl.run()