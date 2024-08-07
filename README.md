# 本地GraphRAG打造AI代码检索助手

## 安装GraphRAG

pip install graphrag

python -m graphrag.index --init --root .#初始化工作空间

## 爬取代码文件

python spider.py

## 进行图谱构建

python -m graphrag.index --root .

python -m graphrag.prompt_tune --root . --no-entity-types#提示调优

## 搭建API和UI界面

python api.py

chainlit run ui.py
