import logging
import os
from pathlib import Path

from utils.dify_client import KnowledgeBaseClient

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

# rules (object): Custom rules (this field is empty in automatic mode)
# pre_processing_rules (array[object]): Pre-processing rules
#   id (string): Unique identifier for the pre-processing rule
#     Enum:
#       remove_extra_spaces: Replace consecutive spaces, newlines, and tabs
#       remove_urls_emails: Remove URLs and email addresses
#   enabled (bool): Indicates whether the rule is selected; if document ID
#   is not provided, it represents the default value
# segmentation (object): Segmentation rules
#   separator: Custom segmentation identifier; currently only one separator
#   is allowed. Default is \n
#   max_tokens: Maximum length (tokens); default is 1000
#   parent_mode: Recall mode for parent segments; options are full-doc
#   (full document recall) / paragraph (paragraph recall)
# subchunk_segmentation (object): Sub-segmentation rules
#   separator: Segmentation identifier; currently only one separator is
#   allowed. Default is ***
#   max_tokens: Maximum length (tokens); must be less than the parent
#   length
#   chunk_overlap: Overlap between segments during segmentation (optional)
DOC_PROCESS_RULE = {
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
            "separator": "\n\n",
            "max_tokens": 500,
            "chunk_overlap": 50
        },
        "parent_mode": "full-doc",
        "subchunk_segmentation": {
            "separator": "\n",
            "max_tokens": 200,
            "chunk_overlap": 0
        }
    }
}

# Initialize logger
logger = logging.getLogger(__name__)


class KBClient:

    def __init__(self, kb_name: str,
                 base_url: str = "https://api.dify.ai/v1"):
        self.kb_name = kb_name
        self.kb_api_key = os.getenv('KB_API_KEY')

        if not self.kb_api_key:
            logger.error("KB_API_KEY not found in environment variables.")
            raise ValueError("KB_API_KEY must be set in environment variables.")

        # First create client without dataset_id to check/create KB
        self.client = KnowledgeBaseClient(api_key=self.kb_api_key,
                                        base_url=base_url)

        # Get or create KB ID
        self.kb_id = self.get_kb(self.kb_name)
        if not self.kb_id:
            response = self.create_kb(self.kb_name)
            self.kb_id = response['id']

        # Recreate client with dataset_id set
        self.client = KnowledgeBaseClient(api_key=self.kb_api_key,
                                        base_url=base_url,
                                        dataset_id=self.kb_id)

    def get_kb(self, name: str) -> str | None:
        """Get the dataset_id of the knowledge base by name.
        Returns None if it does not exist."""
        page = 1
        while True:

            response = self.client.list_datasets(page=page)
            print(response.json())
            has_more = response.json()['has_more']
            kb_list = response.json()['data']
            for kb in kb_list:
                if kb['name'] == name:
                    logger.debug(f"Knowledge base found: {name} "
                                 f"with dataset_id: {kb['id']}")
                    return kb['id']

            if not has_more:
                break

            page += 1

        logger.debug(f"Knowledge base '{name}' does not exist.")
        return None

    def create_or_update_document(
            self,
            document_name: str,
            text: str,
            indexing_technique: str = DOC_DEFAULT_INDEXING_TECHNIQUE,
            doc_form: str = DOC_DEFAULT_FORM,
            doc_language: str = DOC_DEFAULT_LANGUAGE,
            process_rule: dict = DOC_PROCESS_RULE
    ):
        """Create or update a document in the knowledge base using text.

        Args:
            document_name: Name of the document
            text: Content of the document
            indexing_technique: Technique used for indexing, defaults to 
                DOC_DEFAULT_INDEXING_TECHNIQUE
            doc_form: Form of the document, defaults to DOC_DEFAULT_FORM
            doc_language: Language of the document, defaults to DOC_DEFAULT_LANGUAGE
            process_rule: Custom processing rules, defaults to None
        """
        logger.debug(f"Processing document: {document_name}")

        # Search for existing document
        response = self.client.list_documents(keyword=document_name)
        existing_docs = response.json().get('data', [])
        existing_doc = next(
            (doc for doc in existing_docs if doc['name'] == document_name),
            None
        )

        # Prepare extra parameters
        extra_params = {
            'indexing_technique': indexing_technique,
            'doc_form': doc_form,
            'doc_language': doc_language
        }
        if process_rule:
            extra_params['process_rule'] = process_rule

        try:
            if existing_doc:
                # Update existing document
                logger.debug(f"Updating existing document: {document_name}")
                response = self.client.update_document_by_text(
                    document_id=existing_doc['id'],
                    name=document_name,
                    text=text,
                    extra_params=extra_params
                )
                logger.info(f"Updated document: {document_name}")
            else:
                # Create new document
                logger.debug(f"Creating new document: {document_name}")
                response = self.client.create_document_by_text(
                    name=document_name,
                    text=text,
                    extra_params=extra_params
                )
                logger.info(f"Created document: {document_name}")

            logger.debug(f"Response: {response.text}")
            return response

        except Exception as e:
            logger.error(f"Failed to {'update' if existing_doc else 'create'} document: {document_name}. Error: {str(e)}")
            raise

    def create_kb(self, name: str, description: str = None,
                   indexing_technique: str = KB_DEFAULT_INDEXING_TECHNIQUE,
                   permission: str = KB_DEFAULT_PERMISSION,
                   provider: str = KB_DEFAULT_PROVIDER,
                   external_knowledge_api_id: str = None,
                   external_knowledge_id: str = None):
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
        response = self.client.create_dataset(name=name, **payload)
        logger.info(f"Created new knowledge base: {response.text}")
        return response.json()
