"""Conversation management for multi-turn dialogs."""

from ..core.models import ConversationThread, ConversationTurn, Solution
from .llm_interface import BaseLLM


class ConversationManager:
    """Manages multi-turn conversations for solution refinement."""

    def __init__(self, llm: BaseLLM):
        """Initialize conversation manager.

        Args:
            llm: LLM interface for generation
        """
        self.llm = llm
        self.active_conversations: dict[str, ConversationThread] = {}

    def start_conversation(
        self,
        conversation_id: str,
        island_id: int,
        generation: int,
        parent_solutions: list[Solution] | None = None,
    ) -> ConversationThread:
        """Start a new conversation thread.

        Args:
            conversation_id: Unique conversation identifier
            island_id: Island where conversation occurs
            generation: Generation number
            parent_solutions: Parent solutions (if any)

        Returns:
            New conversation thread
        """
        conversation = ConversationThread(
            id=conversation_id,
            island_id=island_id,
            generation=generation,
            parent_solutions=parent_solutions or [],
            children=[],
            turns=[],
        )

        self.active_conversations[conversation_id] = conversation
        return conversation

    def add_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Add a turn to an existing conversation.

        Args:
            conversation_id: Conversation identifier
            turn: Turn to add
        """
        if conversation_id in self.active_conversations:
            conversation = self.active_conversations[conversation_id]
            conversation.turns.append(turn)
            conversation.children.append(turn.generated_solution)
        else:
            raise ValueError(f"Conversation {conversation_id} not found")

    def get_conversation(self, conversation_id: str) -> ConversationThread | None:
        """Get conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation thread or None if not found
        """
        return self.active_conversations.get(conversation_id)

    def end_conversation(self, conversation_id: str) -> ConversationThread | None:
        """End and remove conversation from active list.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Completed conversation thread
        """
        return self.active_conversations.pop(conversation_id, None)

    def get_conversation_history(self, conversation_id: str) -> list[str]:
        """Get formatted conversation history.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of formatted conversation turns
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        history = []
        for turn in conversation.turns:
            if turn.critic_response:
                history.append(
                    f"CRITIC (Turn {turn.turn_number}):\n{turn.critic_response}"
                )
            history.append(f"AUTHOR (Turn {turn.turn_number}):\n{turn.author_response}")

        return history

    def get_active_conversations(self) -> list[str]:
        """Get list of active conversation IDs.

        Returns:
            List of conversation IDs
        """
        return list(self.active_conversations.keys())

    def clear_all_conversations(self) -> None:
        """Clear all active conversations."""
        self.active_conversations.clear()

    def get_conversation_stats(self) -> dict[str, int]:
        """Get statistics about active conversations.

        Returns:
            Dictionary with conversation statistics
        """
        total_conversations = len(self.active_conversations)
        total_turns = sum(
            len(conv.turns) for conv in self.active_conversations.values()
        )
        total_solutions = sum(
            len(conv.children) for conv in self.active_conversations.values()
        )

        return {
            "active_conversations": total_conversations,
            "total_turns": total_turns,
            "total_solutions": total_solutions,
            "avg_turns_per_conversation": total_turns / max(total_conversations, 1),
        }
