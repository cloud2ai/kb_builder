import logging
import os
from pathlib import Path

from kb_builder.utils.dify_client import KnowledgeBaseClient

# Define constants for Knowledge Base (KB) configuration
# Default indexing technique for knowledge bases, valid values:
# high_quanlity / economy
KB_DEFAULT_INDEXING_TECHNIQUE = "high_quality"

# Default permission setting for knowledge bases, valid values:
# only_me / all_team_members / partial_members
KB_DEFAULT_PERMISSION = "only_me"

# Default provider for knowledge bases, valid values:
# vendor: By upload file
# external: External knowledge base
KB_DEFAULT_PROVIDER = "vendor"

# Define constants for Document (DOC) configuration
# Default document indexing technique, valid values:
# high_quality / economy
DOC_DEFAULT_INDEXING_TECHNIQUE = "high_quality"

# Default document form, valid values:
# text_model / hierarchical_model / qa_model
DOC_DEFAULT_FORM = "hierarchical_model"

# Default document language, valid values:
# English / Chinese
DOC_DEFAULT_LANGUAGE = "English"

# Initialize logger
logger = logging.getLogger(__name__)


class KBClient:

    def __init__(self, kb_name: str,
                 base_url: str = "https://api.dify.ai/v1"):
        self.kb_name = kb_name
        self.base_url = base_url
        self.api_key = os.getenv('DIFY_API_KEY')
        if not self.api_key:
            raise ValueError("DIFY_API_KEY environment variable not set")

        # First create client without dataset_id to check/create KB
        self.client = KnowledgeBaseClient(
            api_key=self.api_key,
            base_url=base_url
        )

        # Get or create KB ID
        self.kb_id = self.get_kb(self.kb_name)
        if not self.kb_id:
            response = self.create_kb(self.kb_name)
            self.kb_id = response['id']

        # Recreate client with dataset_id set
        self.client = KnowledgeBaseClient(
            api_key=self.api_key,
            base_url=base_url,
            dataset_id=self.kb_id
        )

    def get_kb(self, name: str) -> str | None:
        """Get the dataset_id of the knowledge base by name.
        Returns None if it does not exist."""
        page = 1
        while True:
            response = self.client.list_datasets(page=page)
            logger.debug(f"Response: {response}")
            has_more = response['has_more']
            kb_list = response['data']
            for kb in kb_list:
                if kb['name'] == name:
                    logger.debug(
                        f"Knowledge base found: {name} "
                        f"with dataset_id: {kb['id']}"
                    )
                    return kb['id']

            if not has_more:
                break

            page += 1

        logger.debug(f"Knowledge base '{name}' does not exist.")
        return None

    def create_or_update_document(
            self,
            document_name: str,
            file_path: str,
            indexing_technique: str = DOC_DEFAULT_INDEXING_TECHNIQUE,
            doc_form: str = DOC_DEFAULT_FORM,
            doc_language: str = DOC_DEFAULT_LANGUAGE,
            process_rule: dict = None,
            parent_separator: str = "\n\n",
            parent_max_tokens: int = 4000,
            parent_chunk_overlap: int = 50,
            parent_mode: str = "paragraph",
            subchunk_separator: str = "\n",
            subchunk_max_tokens: int = 800,
            subchunk_chunk_overlap: int = 0,
            metadata: dict = None
    ):
        """Create or update a document in the knowledge base using a file.

        Args:
            document_name: Name of the document
            file_path: Path to the markdown file
            indexing_technique: Technique used for indexing, defaults to
                DOC_DEFAULT_INDEXING_TECHNIQUE
            doc_form: Form of the document, defaults to DOC_DEFAULT_FORM
            doc_language: Language of the document, defaults to
                DOC_DEFAULT_LANGUAGE
            process_rule: Custom processing rules, defaults to None. If None,
                will use default rules with the following parameters:
            parent_separator: Separator for parent segments, defaults to "##"
            parent_max_tokens: Maximum tokens for parent segments, defaults to
                4000
            parent_chunk_overlap: Overlap between parent segments, defaults to
                50
            parent_mode: Recall mode for parent segments, defaults to
                "paragraph"
            subchunk_separator: Separator for subchunks, defaults to "\n"
            subchunk_max_tokens: Maximum tokens for subchunks, defaults to
                800
            subchunk_chunk_overlap: Overlap between subchunks, defaults to 0
            metadata: Dictionary containing document metadata fields and their
                values. If provided, will be set after document creation/update.
                Supported fields:
                - title (string): Document title
                - background (string): Background information
                - document_summary (string): Document summary
                - menu_name (string): Menu name
                - total_word_count (number): Total word count
        """
        logger.debug(f"Processing document: {document_name}")

        # Search for existing document
        response = self.client.list_documents(keyword=document_name)
        logger.debug(
            f"Get documents with keyword: {document_name} "
            f"response: {response}"
        )
        existing_docs = response.get('data', [])
        existing_doc = next(
            (doc for doc in existing_docs if doc['name'] == document_name),
            None
        )

        # Prepare process rule if not provided
        if process_rule is None:
            process_rule = {
                "mode": "hierarchical",
                "rules": {
                    "pre_processing_rules": [
                        {
                            "id": "remove_extra_spaces",
                            "enabled": True
                        },
                        {
                            "id": "remove_urls_emails",
                            "enabled": False
                        }
                    ],
                    "segmentation": {
                        "separator": parent_separator,
                        "max_tokens": parent_max_tokens,
                        "chunk_overlap": parent_chunk_overlap
                    },
                    "parent_mode": parent_mode,
                    "subchunk_segmentation": {
                        "separator": subchunk_separator,
                        "max_tokens": subchunk_max_tokens,
                        "chunk_overlap": subchunk_chunk_overlap
                    }
                }
            }

        # Prepare document parameters
        doc_params = {
            'name': document_name,
            'indexing_technique': indexing_technique,
            'doc_form': doc_form,
            'process_rule': process_rule
        }

        try:
            if existing_doc:
                # Update existing document
                logger.debug(f"Updating existing document: {document_name}")
                response = self.client.update_document_by_file(
                    document_id=existing_doc['id'],
                    file_path=file_path,
                    extra_params=doc_params
                )
                document_id = existing_doc['id']
                logger.info(f"Updated document: {document_name}")
            else:
                # Create new document
                logger.debug(f"Creating new document: {document_name}")
                response = self.client.create_document_by_file(
                    file_path=file_path,
                    extra_params=doc_params
                )
                document_id = response.get('document', {}).get('id')
                if not document_id:
                    raise ValueError(
                        f"Failed to get document_id from response: "
                        f"{response}"
                    )
                logger.info(f"Created document: {document_name}")

            # Update metadata if provided
            if metadata:
                self.set_document_metadata(document_id, metadata)
                logger.info(
                    f"Updated metadata for document: {document_name}"
                )

            return response

        except Exception as e:
            logger.error(
                f"Failed to {'update' if existing_doc else 'create'} "
                f"document: {document_name}. Error: {str(e)}"
            )
            raise

    def create_kb(
            self,
            name: str,
            description: str = None,
            indexing_technique: str = KB_DEFAULT_INDEXING_TECHNIQUE,
            permission: str = KB_DEFAULT_PERMISSION,
            provider: str = KB_DEFAULT_PROVIDER,
            external_knowledge_api_id: str = None,
            external_knowledge_id: str = None
    ):
        """Create a new knowledge base with specified parameters.

        Default settings recommendations:
        - Visibility: Accessible to all team members.
        - Index Mode: High quality.
        - Search Settings: Hybrid retrieval mode.
        - Weighting:
            - Semantic weight: 0.7
            - Keyword weight: 0.3
        - Top K: 4
        - Score Threshold: 0.4
        """
        payload = {
            "indexing_technique": indexing_technique,
            "permission": permission,
            "provider": provider
        }
        if description:
            payload["description"] = description
        if external_knowledge_api_id:
            payload["external_knowledge_api_id"] = external_knowledge_api_id
        if external_knowledge_id:
            payload["external_knowledge_id"] = external_knowledge_id
        logger.debug(f"Payload: {payload}")
        response = self.client.create_dataset(name, **payload)
        logger.info(f"Created new knowledge base: {response}")
        return response

    def create_kb_metadata(self) -> list:
        """Create default metadata fields for the knowledge base.

        Creates the following metadata fields if they don't exist:
        - title (string): Document title
        - background (string): Background information
        - document_summary (string): Document summary
        - menu_name (string): Menu name
        - total_word_count (number): Total word count

        Returns:
            List of created metadata fields
        """
        # Get existing metadata fields
        existing_metadata = self.get_kb_metadata()
        existing_fields = {
            field['name']: field
            for field in existing_metadata.get('doc_metadata', [])
        }

        metadata_fields = [
            {"type": "string", "name": "title"},
            {"type": "string", "name": "background"},
            {"type": "string", "name": "document_summary"},
            {"type": "string", "name": "menu_name"},
            {"type": "number", "name": "total_word_count"}
        ]

        created_fields = []
        for field in metadata_fields:
            # Skip if field already exists
            if field['name'] in existing_fields:
                logger.info(
                    f"Metadata field {field['name']} already exists, "
                    f"skipping creation"
                )
                continue

            field_response = self.client.add_metadata_field(**field)
            created_fields.append(field_response)
            logger.info(
                f"Created metadata field: {field['name']} "
                f"({field['type']})"
            )

        return created_fields

    def get_kb_metadata(self) -> dict:
        """Get all metadata fields for the knowledge base.

        Returns:
            Dictionary containing:
            - doc_metadata: List of metadata fields
            - built_in_field_enabled: Whether built-in fields are enabled
        """
        response = self.client.list_metadata_fields()
        return response

    def set_document_metadata(
            self,
            document_id: str,
            metadata: dict
    ) -> dict:
        """Set metadata for a document.

        Args:
            document_id: ID of the document to set metadata for
            metadata: Dictionary containing metadata fields and their values.
                     Supported fields:
                     - title (string): Document title
                     - background (string): Background information
                     - document_summary (string): Document summary
                     - menu_name (string): Menu name
                     - total_word_count (number): Total word count

        Returns:
            Dictionary containing the updated document metadata

        Raises:
            ValueError: If metadata contains unsupported fields
        """
        # Validate metadata fields
        supported_fields = {
            "title", "background", "document_summary",
            "menu_name", "total_word_count"
        }
        invalid_fields = set(metadata.keys()) - supported_fields
        if invalid_fields:
            raise ValueError(
                f"Unsupported metadata fields: {invalid_fields}. "
                f"Supported fields are: {supported_fields}"
            )

        # Get existing metadata fields to get their IDs
        metadata_fields = self.get_kb_metadata()
        field_id_map = {
            field['name']: field['id']
            for field in metadata_fields.get('doc_metadata', [])
        }

        # Convert metadata to the required format
        operation_data = [{
            "document_id": document_id,
            "metadata_list": [
                {
                    "id": field_id_map[field_name],
                    "name": field_name,
                    "value": field_value
                }
                for field_name, field_value in metadata.items()
                if field_name in field_id_map
            ]
        }]

        response = self.client.update_document_metadata(
            operation_data=operation_data
        )

        logger.info(
            f"Successfully set metadata for document {document_id}"
        )
        return response

    def check_document_exists(self, doc_name: str) -> bool:
        """
        Check if document exists in the knowledge base.

        Args:
            doc_name: Name of the document to check

        Returns:
            bool: True if document exists, False otherwise
        """
        response = self.client.list_documents(keyword=doc_name)
        docs = response.get('data', [])
        return len(docs) > 0

    def check_content_unchanged(
        self,
        doc_name: str,
        current_content: str,
        uploaded_path: Path
    ) -> bool:
        """
        Check if document content has changed.

        Args:
            doc_name: Name of the document
            current_content: Current content of the document
            uploaded_path: Path to the previously uploaded content

        Returns:
            bool: True if content is unchanged, False otherwise
        """
        if not uploaded_path.exists():
            return False

        with open(uploaded_path, 'r', encoding='utf-8') as f:
            uploaded_content = f.read()
        return current_content == uploaded_content

    def upload_document(
        self,
        doc_name: str,
        file_path: str,
        lang: str,
        metadata: dict = None
    ) -> bool:
        """
        Upload document to knowledge base.

        Args:
            doc_name: Name of the document
            file_path: Path to the document file
            lang: Language code
            metadata: Optional metadata for the document

        Returns:
            bool: True if upload was successful
        """
        try:
            doc_language = "English" if lang == "en" else "Chinese"
            self.create_or_update_document(
                document_name=doc_name,
                file_path=file_path,
                doc_language=doc_language,
                metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload document {doc_name}: {e}")
            raise

    def save_uploaded_content(
        self,
        content: str,
        uploaded_path: Path
    ) -> None:
        """
        Save uploaded content to local file.

        Args:
            content: Content to save
            uploaded_path: Path to save the content
        """
        with open(uploaded_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def cleanup_documents(self, processed_files: list[str]) -> None:
        """
        Clean up documents that no longer exist in the source directory.
        This includes:
        1. Getting all documents from the knowledge base
        2. Comparing with local processed files
        3. Deleting documents that only exist in KB but not in local files

        Args:
            processed_files: List of processed file names (without .txt extension)
        """
        logger.info("Starting document cleanup process")

        try:
            # Get all documents from knowledge base
            response = self.client.list_documents(get_all=True)
            kb_docs = response.get('data', [])
            logger.debug(f"Found {len(kb_docs)} documents in KB")

            # Build a mapping from document name (without .txt) to document id
            kb_doc_mappings = {
                doc['name'].replace('.txt', ''): doc['id']
                for doc in kb_docs
            }
            kb_doc_names = set(kb_doc_mappings.keys())

            # Convert processed_files to set for comparison
            processed_files_set = set(processed_files)

            # Find documents that only exist in KB but not in local files
            docs_to_delete = kb_doc_names - processed_files_set

            if docs_to_delete:
                logger.info(
                    f"Found {len(docs_to_delete)} documents to delete: "
                    f"{', '.join(docs_to_delete)}"
                )

                # Delete documents from KB
                for doc_name in docs_to_delete:
                    logger.info(
                        f"Deleting document from KB: {doc_name}"
                    )
                    try:
                        doc_id = kb_doc_mappings[doc_name]
                        self.client.delete_document(doc_id)
                        logger.info(
                            f"Successfully deleted document: {doc_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error deleting document {doc_name}: {str(e)}"
                        )
            else:
                logger.info("No documents need to be deleted")

            logger.info(
                f"Cleanup completed: {len(docs_to_delete)} documents deleted"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
