from pymilvus import MilvusClient, DataType


class VectorDB:
    client = None
    db_name = "default"
    collection_name = "properties"
    milvus_uri = "http://localhost:19530"

    def __init__(self, delete_collection=False):
        self.client = MilvusClient(uri=self.milvus_uri, db_name=self.db_name)
        if delete_collection:
            self.client.drop_collection(
                collection_name=self.collection_name
            )
        if not self.client.has_collection(self.collection_name):
            self.create_collection()

    def create_collection(self):
        schema = MilvusClient.create_schema()

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="bedrooms", datatype=DataType.INT8)
        schema.add_field(field_name="bathrooms", datatype=DataType.INT8)
        schema.add_field(field_name="carspaces", datatype=DataType.INT8)
        schema.add_field(field_name="floor", datatype=DataType.INT8)
        schema.add_field(field_name="area", datatype=DataType.FLOAT)
        schema.add_field(field_name="price", datatype=DataType.FLOAT)
        schema.add_field(field_name="province", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="city", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="district", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="build_year", datatype=DataType.INT16)
        schema.add_field(field_name="list_at", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="decoration", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="distance_to_metro", datatype=DataType.FLOAT)
        schema.add_field(field_name="distance_to_school", datatype=DataType.FLOAT)
        schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="desc_vector", datatype=DataType.FLOAT_VECTOR, dim=384)

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="desc_vector",
            index_name="desc_vector_index",
            index_type="FLAT",  # https://milvus.io/docs/zh/index-explained.md
            metric_type="L2"  # 度量类型，有多种可以对比效果 https://milvus.io/docs/zh/metric.md，可以考虑使用多种度量类型分别搜索，综合排序
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def upsert(self, data):
        self.client.upsert(
            collection_name=self.collection_name,
            data=data
        )

    def search(self, vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            top_k=top_k,
            metric_type="L2"
        )
        return results
