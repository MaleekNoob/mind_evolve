"""Feedback generation utilities."""

from typing import Any, Dict, List

from ..core.models import EvaluationResult, Problem, Solution


class FeedbackGenerator:
    """Generates textual feedback based on evaluation results."""
    
    def __init__(self, feedback_style: str = "constructive"):
        """Initialize feedback generator.
        
        Args:
            feedback_style: Style of feedback (constructive, detailed, concise)
        """
        self.feedback_style = feedback_style
        
    def generate_feedback(self, 
                         solution: Solution,
                         problem: Problem,
                         evaluation: EvaluationResult) -> List[str]:
        """Generate comprehensive feedback for a solution.
        
        Args:
            solution: Solution that was evaluated
            problem: Original problem
            evaluation: Evaluation results
            
        Returns:
            List of feedback messages
        """
        feedback = []
        
        # Score-based feedback
        if evaluation.score < 3.0:
            feedback.append("Solution needs significant improvement")
        elif evaluation.score < 7.0:
            feedback.append("Solution shows promise but has room for improvement")
        else:
            feedback.append("Solution demonstrates good quality")
            
        # Constraint violation feedback
        if evaluation.constraint_violations > 0:
            feedback.append(f"Solution violates {evaluation.constraint_violations} constraint(s)")
            feedback.extend(self._generate_constraint_feedback(problem, evaluation))
        else:
            feedback.append("All constraints appear to be satisfied")
            
        # Content quality feedback
        content_feedback = self._generate_content_feedback(solution.content)
        feedback.extend(content_feedback)
        
        # Improvement suggestions
        if evaluation.score < 8.0:
            suggestions = self._generate_improvement_suggestions(solution, problem, evaluation)
            feedback.extend(suggestions)
            
        return feedback
        
    def _generate_constraint_feedback(self, 
                                    problem: Problem,
                                    evaluation: EvaluationResult) -> List[str]:
        """Generate specific feedback about constraint violations.
        
        Args:
            problem: Problem definition
            evaluation: Evaluation results
            
        Returns:
            List of constraint-specific feedback
        """
        # Extract constraint violations from evaluation feedback
        constraint_feedback = [
            msg for msg in evaluation.feedback 
            if "constraint" in msg.lower() and "violated" in msg.lower()
        ]
        
        if not constraint_feedback:
            constraint_feedback = ["Some constraints may not be fully addressed"]
            
        return constraint_feedback
        
    def _generate_content_feedback(self, content: str) -> List[str]:
        """Generate feedback about solution content quality.
        
        Args:
            content: Solution content
            
        Returns:
            List of content-specific feedback
        """
        feedback = []
        
        word_count = len(content.split())
        line_count = len(content.strip().split('\n'))
        
        # Length feedback
        if word_count < 20:
            feedback.append("Solution may be too brief - consider adding more detail")
        elif word_count > 500:
            feedback.append("Solution may be too verbose - consider being more concise")
            
        # Structure feedback
        if line_count == 1:
            feedback.append("Consider organizing solution with better structure (paragraphs, lists)")
        elif any(line.strip().startswith(('-', '*', '1.', '2.')) for line in content.split('\n')):
            feedback.append("Good use of structured formatting")
            
        # Specificity feedback
        if content.count(',') < 2 and word_count > 30:
            feedback.append("Solution could benefit from more specific details")
            
        return feedback
        
    def _generate_improvement_suggestions(self, 
                                        solution: Solution,
                                        problem: Problem, 
                                        evaluation: EvaluationResult) -> List[str]:
        """Generate specific improvement suggestions.
        
        Args:
            solution: Current solution
            problem: Problem definition
            evaluation: Evaluation results
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Score-based suggestions
        if evaluation.score < 5.0:
            suggestions.append("Consider completely restructuring your approach")
            suggestions.append("Review the problem requirements more carefully")
        else:
            suggestions.append("Focus on addressing specific constraint violations")
            suggestions.append("Consider adding more detail to strengthen your solution")
            
        # Constraint-based suggestions
        if evaluation.constraint_violations > 0:
            suggestions.append("Review each constraint and ensure your solution explicitly addresses them")
            suggestions.append("Consider adding explanations for how you satisfy each requirement")
            
        # Content-based suggestions
        if len(solution.content.split()) < 50:
            suggestions.append("Expand your solution with more comprehensive details")
        
        # Problem-specific suggestions
        if problem.examples and "example" not in solution.content.lower():
            suggestions.append("Consider referencing or building upon the provided examples")
            
        return suggestions[:3]  # Limit to top 3 suggestions
        
    def format_feedback_for_display(self, feedback: List[str]) -> str:
        """Format feedback list for display.
        
        Args:
            feedback: List of feedback messages
            
        Returns:
            Formatted feedback string
        """
        if not feedback:
            return "No specific feedback available."
            
        if self.feedback_style == "concise":
            return "; ".join(feedback[:3])
        elif self.feedback_style == "detailed":
            formatted = []
            for i, msg in enumerate(feedback, 1):
                formatted.append(f"{i}. {msg}")
            return "\n".join(formatted)
        else:  # constructive
            return "\n".join(f"• {msg}" for msg in feedback)
            
    def get_feedback_summary(self, 
                           solutions: List[Solution],
                           evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary of feedback across multiple solutions.
        
        Args:
            solutions: List of solutions
            evaluations: List of evaluation results
            
        Returns:
            Dictionary with feedback summary statistics
        """
        if not solutions or not evaluations:
            return {}
            
        # Aggregate statistics
        total_feedback_items = sum(len(eval_result.feedback) for eval_result in evaluations)
        avg_score = sum(eval_result.score for eval_result in evaluations) / len(evaluations)
        total_violations = sum(eval_result.constraint_violations for eval_result in evaluations)
        
        # Common feedback themes
        all_feedback = []
        for eval_result in evaluations:
            all_feedback.extend(eval_result.feedback)
            
        # Find most common feedback keywords
        common_keywords = self._extract_common_keywords(all_feedback)
        
        return {
            "total_solutions": len(solutions),
            "avg_score": round(avg_score, 2),
            "total_feedback_items": total_feedback_items,
            "total_violations": total_violations,
            "avg_violations_per_solution": round(total_violations / len(solutions), 2),
            "common_issues": common_keywords[:5],  # Top 5 common issues
        }
        
    def _extract_common_keywords(self, feedback_list: List[str]) -> List[str]:
        """Extract common keywords from feedback messages.
        
        Args:
            feedback_list: List of feedback messages
            
        Returns:
            List of common keywords sorted by frequency
        """
        from collections import Counter
        import re
        
        # Extract meaningful words
        all_words = []
        for feedback in feedback_list:
            words = re.findall(r'\b\w{4,}\b', feedback.lower())  # Words with 4+ chars
            all_words.extend(words)
            
        # Count frequency and return most common
        word_counts = Counter(all_words)
        
        # Filter out common stop words
        stop_words = {'solution', 'constraint', 'problem', 'should', 'could', 'would', 'consider'}
        filtered_counts = {word: count for word, count in word_counts.items() 
                          if word not in stop_words}
        
        return [word for word, _ in Counter(filtered_counts).most_common()]