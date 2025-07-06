# ------------------ compatibility aliases for legacy agent demos ------------------
# Older agent demos expect wrapper classes named SearchEmails, ReadEmail, AnswerQuestion, Terminate.
# Provide aliases to the new tool classes to maintain backward compatibility.
SearchEmails = SearchEmailsTool  # type: ignore
ReadEmail = ReadEmailTool        # type: ignore
AnswerQuestion = AnswerQuestionTool  # type: ignore
Terminate = TerminateTool        # type: ignore

# ------------------ compatibility aliases for legacy agent demos ------------------
SearchEmails = SearchEmailsTool  # type: ignore
ReadEmail = ReadEmailTool  # type: ignore
AnswerQuestion = AnswerQuestionTool  # type: ignore
Terminate = TerminateTool  # type: ignore 