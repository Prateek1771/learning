We'll start with **Project 1: Context Window & Working Memory Simulator**. I'll use a real-world coding assistant as our running example so every concept has a practical purpose.

# **Project 1 — Build an AI Coding Assistant with Working Memory**



Beginner → Intermediate

Python + LLM



## **1. Real-world scenario: AI coding assistant**

Imagine building a simplified version of Cursor or GitHub Copilot.

A developer is working on an application and interacts with your AI assistant:

**DevMate — AI Coding Assistant**

Example session

I'm building an e-commerce application using React and FastAPI.

Understood. You're building an e-commerce application using React and FastAPI.

My database is PostgreSQL and I'm using JWT authentication.

Got it. PostgreSQL for data storage and JWT for authentication.

Create an API endpoint to add products to my cart.

I'll create a FastAPI endpoint using your PostgreSQL database and JWT authentication.

Now implement the checkout API.

I'll continue using the same architecture and authentication approach.

The assistant uses earlier messages to understand the current task.

Now imagine the conversation continues for 100 messages.

The developer has discussed:

- Database schema.
- API endpoints.
- Authentication.
- Payment integration.
- Error handling.
- Deployment.

But your LLM has a limited context window.

**What happens when the conversation becomes too large to fit into the model's context?**

That's the problem we're solving.

# **2. Understand the core concepts first**

### **Concept 1: What is a context window?**

A context window is the maximum amount of information a model can process in a single interaction.

It generally includes:

- System instructions.
- Conversation history.
- Retrieved documents.
- Tool outputs.
- Current user message.
- Space required for the model's response, depending on the API's token limits.

Think of it as RAM available to the model for one inference.

[Chapter 2: Core Concepts and Terminology | Prompting Handbook](https://images.openai.com/static-rsc-4/9aaQeHpEFDp56cXFacZQ6BHsuwx-QW2KqTrdp3jyKaTvh2QOSRkUFPbpnsxY6v5YpVaRg1CIubWIjCfO21PfRkz2eDGKVyyweRclY0FLFK1PNEr9uocnjnAgEU9QCvd7dCQ_qNppG69hU0ib5XErM262GN_4KXwwRynNup8BKCY?purpose=inline)

[zettai-seigi.github.io](https://zettai-seigi.github.io/StructuredPromptingHandbook/chapters/02-core-concepts/?utm_source=chatgpt.com)

Example:

Suppose our model has a context budget of 8,000 tokens.

| Component                | Tokens    |
| ------------------------ | --------- |
| System prompt            | 500       |
| Conversation history     | 5,000     |
| Tool outputs             | 1,000     |
| Current question         | 500       |
| Reserved response budget | 1,000     |
| **Total**                | **8,000** |

If the next conversation requires more tokens, we need a strategy to manage the available context.

Important: context windows are model-specific. These numbers are illustrative, not actual limits for a particular model.

### **Concept 2: What is a token?**

A token is a unit of text processed by an LLM.

It can represent:

- A complete word.
- Part of a word.
- Punctuation.
- A space or other text unit.

For example:

```
"PostgreSQL is a database"
```

Might be tokenized into several individual tokens.

Why does this matter?

Because we cannot reliably manage context by counting characters or words. We need token-based budgeting.

### **Concept 3: Working Memory**

Working memory is information the agent actively maintains while performing its current task.

In our coding assistant:

```
Current project:
  Name: E-commerce

Current task:
  Implement checkout API

Relevant context:
  Frontend: React
  Backend: FastAPI
  Database: PostgreSQL
  Authentication: JWT

Recent conversation:
  Last 5 messages
```

This information is what the agent needs to complete the current task.

Working memory is not necessarily the same as the entire conversation history.

# **3. The six memory strategies we'll implement**

We'll build the same coding assistant with six different strategies and compare their behavior.

**Strategy A — Full Conversation Buffer**

[How We Built Chat Memory That Actually Works — Lessons from Shipping to 100K Users - DEV Community](https://images.openai.com/static-rsc-4/pJ7uOs-qad4bL8gZAmPlUySix_90MBt3tfTvS7B_tbEfe74TVIAuG8GnND4axNb3ijYCnNDnFPBu7Xz2VerOedr6lYe3RoEPIcAv2tTeNYm59HB96yh2uSfO5NBuXqh7MXZauuFN2yvJCS_SLbFaqx3yoR3irNpXifN6TuhJDuI?purpose=inline)

[dev.to](https://dev.to/mrbolo/how-we-built-chat-memory-that-actually-works-lessons-from-shipping-to-100k-users-24cd?utm_source=chatgpt.com)

Stores every message and sends the complete history to the model.

Problem: eventually exceeds the context budget.

**Strategy B — Sliding Window Memory**

[The Three Memory Types Every LLM Developer Must Know | by Sahil Nanga | Medium](https://images.openai.com/static-rsc-4/weOFLyXWPLq8duNCxGc_LalhCGu2xOfvZ-csnzo1uf83kguKdHEINQ2NQ9bmfTMGaWftMFqXkHzK67ILPhXaoOVDB8WzWYLVRVspmXus6AqCOpMskxgFH9w7mqSNVxWKvSjVTOoZMzFBWildHiC5TpFH4SgCsnEPqSctJRNefk4?purpose=inline)

[medium.com](https://medium.com/%40sahilnanga4/the-three-memory-types-every-llm-developer-must-know-3358c26fdff3?utm_source=chatgpt.com)

Keeps only the latest N messages.

Problem: important older information disappears.

**Strategy C — FIFO Memory**

[Complete DSA Series Part 8: Queue - Wasil Zafar](https://images.openai.com/static-rsc-4/UzbgUG0OECeVwOYzWMPtBDTqj1CmsdRapN1h0bBIvivz9zCJykWeJep6Np_EDF8vW87V_-z7FHNr7QQTd2su0HbMk19VneUdEFr32GtyN5NAqdbzPDLDGedMHZ0Vm8CAl04Yl3BbVley5ylPi5PmusdG_n4HyhtXi-4m6-6T4XY?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/data-structures/dsa-queue-implementations.html?utm_source=chatgpt.com)

Removes the oldest message when the memory capacity is reached.

Problem: the oldest message might contain critical project information.

**Strategy D — Summarization Memory**

[Context Management for Deep Agents](https://images.openai.com/static-rsc-4/aqSFzlZfWhfvp3ouGKWSIT-1BRUjtN-90N5vawnl35c1dlDn65vl38ZCMRWv0nPblkTYEE2z_W-5XGgi3esXQ5D7QH7S6r6WRSqJlFB2fiWZZGw2iQ30sSU_I9hU5eJofqR4SqOjKyPXhlA8QKe14RQiEulMHC8htZI_HtIhdOY?purpose=inline)

[blog.langchain.com](https://blog.langchain.com/context-management-for-deepagents/?utm_source=chatgpt.com)

Compresses older messages into a summary.

Problem: summarization can lose technical details.

**Strategy E — Priority-Based Memory**

Assigns importance scores to messages.

Example:

- Database choice: high priority.
- Authentication method: high priority.
- Casual greeting: low priority.

Problem: scoring importance accurately is difficult.

**Strategy F — Hybrid Working Memory**

Combines:

- Recent conversation messages.
- Important project facts.
- Older conversation summary.
- Current task information.

This will be our final implementation.

Target architecture

# **4. Architecture we'll build**

[Lesson 13: Introduction to Context Engineering](https://images.openai.com/static-rsc-4/rIyJBJLL_aJywZAyAG5MAVLdMacYNYtPaNtSUBD3Z8jxYVjtkeH5bi7Qtk8f3A-n98bN9MZn7jZHk_5q4AInWgk0x0xAhb9zsUsr3QAuwNRTy4WDjX_mTjCSeprFfAHquGdTkROZj2rjKLhSUP8ZSHfk_JBDGxzjO2H72bZJspc?purpose=inline)

[aiamastery.substack.com](https://aiamastery.substack.com/p/lesson-13-introduction-to-context?utm_source=chatgpt.com)

Our execution flow:

```
              User Message
                   |
                   v
           Conversation Manager
                   |
                   v
             Memory Manager
                   |
          +--------+--------+
          |                 |
          v                 v
     Token Counter    Memory Strategy
          |                 |
          +--------+--------+
                   |
                   v
             Context Builder
                   |
                   v
                 LLM
                   |
                   v
             Agent Response
                   |
                   v
           Update Conversation
```

We'll implement this architecture in Python.

# **5. Technology stack**

| Technology           | Purpose                     |
| -------------------- | --------------------------- |
| Python 3.12+         | Core implementation         |
| FastAPI              | Optional API layer          |
| OpenAI or Gemini API | LLM inference               |
| tiktoken             | Token counting              |
| Pydantic             | Data validation             |
| SQLite               | Persist conversations later |

We'll start with Python and tiktoken. We'll introduce the LLM API after understanding the memory manager.

# **6. Implementation roadmap**

We'll split Project 1 into five milestones.

| Milestone | What we build                    | Outcome                             |
| --------- | -------------------------------- | ----------------------------------- |
| 1         | Token counter                    | Understand tokenization and budgets |
| 2         | Conversation buffer              | Store messages                      |
| 3         | Memory strategies                | FIFO, sliding window                |
| 4         | Summarization and prioritization | Intelligent memory selection        |
| 5         | Hybrid working memory            | Complete coding assistant           |

## **7. Milestone 1 — Build a token counter**

Our first coding task

Before building an intelligent memory system, we need to measure how much information we're storing.

**Your task:** Build a Python program that:

1. Accepts a conversation as input.
2. Counts tokens for each message.
3. Calculates total conversation tokens.
4. Accepts a maximum token budget.
5. Shows how much context capacity remains.

### **Step 1: Set up the project**

Create this directory:

```
project-01-working-memory/
│
├── main.py
├── requirements.txt
└── README.md
```

Install the dependency:

```
pip install tiktoken
```

### **Step 2: Understand the tokenizer**

We'll use `tiktoken`, a tokenizer library commonly used with OpenAI models.

Try this:

```
import tiktokenencoding = tiktoken.get_encoding("cl100k_base")text = "PostgreSQL is my preferred database"tokens = encoding.encode(text)print("Tokens:", tokens)print("Token count:", len(tokens))
```

Understand these two operations:

- `encode()` converts text into token IDs.
- `len()` counts the resulting tokens.

Note: `cl100k_base` is a particular tokenizer encoding, not a universal tokenizer for all LLMs.

### **Step 3: Your first assignment**

Implement this function yourself:

```
def count_tokens(text: str) -> int:    # Your implementation here    pass
```

Then implement:

```
def calculate_remaining_tokens(    messages,    max_tokens,    reserved_tokens):    # Your implementation here    pass
```

Expected behavior:

**Example output**

```
Context Budget: 8000 tokens

System Prompt:     500
Conversation:     3500
Current Query:     200
Reserved Output:  1000

---------------------------
Total Used:       5200
Remaining:        2800
Utilization:       65%
```

Expected result

Illustrative numbers for your implementation.

# **8. Test your understanding**

Before moving to Milestone 2, make sure you can answer these questions.

**Knowledge checkpoint**

0/4 answered

**1. What happens when an LLM receives more input tokens than its context window allows?**

It automatically creates another context window

The request must be reduced or rejected depending on the API

It automatically remembers everything

**2. Why should we count tokens instead of characters?**

Tokens are what LLM context limits are measured in

Characters always equal tokens

Tokens are only needed for embeddings

**3. Which strategy removes the oldest messages first?**

Summarization

FIFO

Priority-based memory

**4. What is the biggest disadvantage of a simple sliding window?**

It requires a graph database

It cannot store strings

It may discard important older information

Check answers

## **What I want you to do now**

Complete only Milestone 1.

When you're done, share your `main.py` implementation here. I'll review it like a code reviewer:

- Correctness of token counting.
- Edge cases.
- Function design.
- Token budget calculations.
- Potential production improvements.

**Don't move to summarization or vector databases yet.** First understand why an agent needs working memory and how context is managed.

# **Project 1 — DevMate: AI Working Memory & Context Management Engine**



Project 01

Beginner → Intermediate

Hands-on engineering



We're going to build a simplified AI coding assistant that can manage long conversations without blindly sending the entire conversation history to an LLM.

The goal isn't just to build a chatbot. The goal is to understand and implement the **memory management layer behind an AI agent**.

By completing this project, you'll understand how context windows work, why agents lose information, how memory gets compressed, how to decide what to retain, and how to construct context for an LLM.

# **1. Real-world problem statement**

Imagine you're building an AI coding assistant similar to Cursor.

A developer starts a conversation:

**DevMate — AI Coding Assistant**

Demo scenario

**Developer — Message 1**

I'm building an e-commerce application using React, FastAPI and PostgreSQL.

**DevMate**

Understood. I'll use React for the frontend, FastAPI for the backend and PostgreSQL for the database.

**Developer — Message 2**

I'm using JWT authentication and Redis for caching.

**DevMate**

Got it. Your application uses JWT authentication and Redis caching.

**Developer — Message 3**

Help me implement the cart API.

**DevMate**

I'll implement the cart API using FastAPI, PostgreSQL and your JWT authentication setup.

**Developer — Message 100**

Now implement the checkout API. Use the same architecture we discussed earlier.

**The problem**

The conversation has become extremely long. Sending all 100 messages repeatedly is inefficient and may exceed the model's context budget.

Your job is to make DevMate intelligent enough to manage its working context.

It should:

- Remember recent messages.
- Preserve important project information.
- Compress older conversations.
- Remove less useful information when necessary.
- Track its token budget.
- Reconstruct the correct context for every new request.

# **2. What exactly are we building?**

[短期记忆 + 长期记忆 + Context 压缩，让有了 Memory 的 Agent 不再失忆。 - 技术派 - Java技术社区 | RAG+Agent实战项目教程+AI助手](https://images.openai.com/static-rsc-4/Ap2cJqscjyHTC8WKkfjngrH5OCcw0-6caNkKPCBdctD8Ygw01XfUikJgOGYVwBiM-ldJkmMqAPYMIPvpwvRn31qSIep5XClvxyXlfDhfpR3B4S43Fjv-nrnEfk191u7ZIzyqfYYKSb5_PLez8f67zy1hGKeYDVvRZVHvl4Ijxvg?purpose=inline)

[paicoding.com](https://paicoding.com/memory-system?utm_source=chatgpt.com)

Our application will have five primary components.

| Component              | Responsibility                              |
| ---------------------- | ------------------------------------------- |
| Conversation Manager   | Maintains user and assistant messages       |
| Token Manager          | Counts tokens and tracks the context budget |
| Working Memory Manager | Selects which messages stay in context      |
| Summarization Engine   | Compresses older conversations              |
| Context Builder        | Constructs the final prompt sent to the LLM |

### **Expected application behavior**

Suppose the developer has exchanged 100 messages.

Instead of sending all 100 messages, DevMate constructs something like:

```
SYSTEM INSTRUCTIONS
-------------------
You are an AI coding assistant.

PROJECT SUMMARY
---------------
Project: E-commerce application
Frontend: React
Backend: FastAPI
Database: PostgreSQL
Authentication: JWT
Cache: Redis

PREVIOUS PROGRESS
-----------------
Product API completed.
Cart API completed.
Payment integration pending.

RECENT CONVERSATION
-------------------
Last 5 relevant messages.

CURRENT REQUEST
---------------
Implement the checkout API.
```

The model receives a compact representation containing the information needed for the current task.

This is the core of working memory management.

# **3. Concepts you will learn**

This project will cover the following concepts in depth.

## **A. LLM context fundamentals**

| Concept             | What you need to understand                       |
| ------------------- | ------------------------------------------------- |
| Context window      | Maximum tokens supported in a model context       |
| Tokenization        | How text becomes tokens                           |
| Input tokens        | Tokens supplied to the model                      |
| Output tokens       | Tokens generated by the model                     |
| Token budget        | Allocated tokens for different context components |
| Context overflow    | What happens when limits are exceeded             |
| Context utilization | Percentage of available context being used        |

## **B. Working memory strategies**

You'll implement each strategy independently.

| Strategy           | Implementation                                         |
| ------------------ | ------------------------------------------------------ |
| Buffer memory      | Store the complete conversation                        |
| FIFO memory        | Remove oldest messages first                           |
| Sliding window     | Keep the latest N messages                             |
| Token-based window | Keep messages within a token budget                    |
| Summary memory     | Compress older messages                                |
| Priority memory    | Preserve important messages                            |
| Hybrid memory      | Combine summaries, important facts and recent messages |

## **C. Summarization techniques**

This is particularly important based on your earlier question.

You'll implement:

1. Full conversation summarization.
2. Rolling summarization.
3. Incremental summarization.
4. Summary updates.
5. Summary compression.
6. Summary preservation across sessions.
7. Summary quality evaluation.

We'll also study what happens when a summary itself becomes too large.

## **D. Memory management algorithms**

- FIFO eviction.
- Sliding window.
- Priority-based eviction.
- Recency-based selection.
- Importance scoring.
- Token-aware selection.
- Context compression.
- Memory truncation.

## **E. Production considerations**

- Prompt injection through conversation history.
- Preserving important instructions.
- Handling tool calls and tool outputs.
- Avoiding duplicate summaries.
- Preventing summary drift.
- Model-specific tokenization.
- Latency and LLM costs.
- Context reconstruction after application restart.

# **4. Technology stack**

These are the tools we'll use.

[Part II: AI — Chapter 3.1: Familiarisation with Python | Solutions for Class 9 ICSE Exploring Robotics & Artificial Intelligence | KnowledgeBoat](https://images.openai.com/static-rsc-4/-e04Bxey6vqUTFrO88fANCW1nPXeVIFiPNJeo6Kw8yPQZ8Q41SilqpodpSdyFJhWy2EVXNyL1Xn_4x8Hnn34Rk2Hv8hJgvDJ1ojmNcpBGJ90THGWn9ocNm2NADt_jIt0C2obkoNJmYls0vCArCmkfyXoXKalMJnhwArBGP-BUV8?purpose=inline)

**Python**

Core implementation language

Required

[Chatte mit den besten KI-Modellen der Welt.](https://images.openai.com/static-rsc-4/5WMgkKYZPRc8FKemY4U2q-9SgbBfKC6LnjgRJhp12FkyjqYQq4WTU8kliZxAQUobXMJLDXGnT0xo0D9bj1RUhAZeS-AFAyZmL4-OKw_eCN2bGTcubYYvlMYXbichcH6tDbjOAkp088W09tpvel4Hvv9kgFV3iKobqn5tdJksZZU?purpose=inline)

**OpenAI API / Gemini API**

Conversation generation and summarization

One required

[GitHub - openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI's models. · GitHub](https://images.openai.com/static-rsc-4/BfFoHgsP62Cdfiudy1VPD3detFbdiTWFekVcMF54cW-2ys46yokTsl86TLL8wKXGwOrrCFdBep5wjraTAaMOiBF217-AiH6GCg9J3DwslFmJ-EDEwZ2VWJ9J9U3cHFPHazm1Ew4Q7lkgL1w1gfsJtcWRSf53mzwg9XTCvlkh2k0?purpose=inline)

**tiktoken**

Token counting and context budgeting

Required

[How to Keep LLM Outputs Predictable Using Pydantic Validation](https://images.openai.com/static-rsc-4/C3qqGR_uIO9EMFF7hfoe4lmQ3Nim0mL6gJN9lsMt2uZjI0ztdpPQTp_C8Iv-wQ4gg8oxk6s51gtbQaOephcsQO63SbSsKS5wNeNtKUuMxGxVEgp9BUBcgtPFsllUJkU39AZcichC9SH4r6pAtPhX9sl2C-d4h0sVaqYUuv4L0J8?purpose=inline)

**Pydantic**

Structured message and memory models

Required

[Logo SQLite – Logos PNG](https://images.openai.com/static-rsc-4/DB8WVS6NrLCtUSYIc2zTPGzrMvrfjPYZpaPLLiELKaGa-8ig6i55NjodHWZqWiL__4SWevhxL3ba1kAF3Oq8brovMP1dXTdtm6k7QmtOMoYPoCrfB7byzg0w9qPHn11PT6I-8w3z6cczqMbyPlGjM0DKm0LT-4oa8zA_FnqDFow?purpose=inline)

**SQLite**

Conversation persistence

Later milestone

[FastAPI Training online | Certification Course](https://images.openai.com/static-rsc-4/3oqMqrp7oPRSriMt9xXVBHZHsPei8fEWuA-8L4K7XuJ6mr_sRjvu-9xG8XhwQ6Cxii8kMDYBf2IPpic0w9lsd_WWzuUoqZsnVObn85H1cUBOsBkrUM-hiHgqvF7zNkL7ioxWOvl6qzqnTZgxAN23QuSS09nFm6DaRwaX6wcibpI?purpose=inline)

**FastAPI**

Expose the memory engine through APIs

Optional

Useful documentation:

- [tiktoken — GitHub](https://github.com/openai/tiktoken?utm_source=chatgpt.com)
- [Python documentation](https://docs.python.org/3/?utm_source=chatgpt.com)
- [OpenAI conversation state](https://platform.openai.com/docs/guides/conversation-state?utm_source=chatgpt.com)
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs?utm_source=chatgpt.com)
- [Pydantic documentation](https://docs.pydantic.dev/latest/?utm_source=chatgpt.com)

**My recommendation:** Use Python, tiktoken and SQLite initially. Add an LLM API when we reach summarization. This lets you understand the underlying algorithms before relying on an LLM to do the work.

# **5. Project architecture**

Here is the architecture we will implement progressively.

[Lesson 13: Introduction to Context Engineering](https://images.openai.com/static-rsc-4/rIyJBJLL_aJywZAyAG5MAVLdMacYNYtPaNtSUBD3Z8jxYVjtkeH5bi7Qtk8f3A-n98bN9MZn7jZHk_5q4AInWgk0x0xAhb9zsUsr3QAuwNRTy4WDjX_mTjCSeprFfAHquGdTkROZj2rjKLhSUP8ZSHfk_JBDGxzjO2H72bZJspc?purpose=inline)

[aiamastery.substack.com](https://aiamastery.substack.com/p/lesson-13-introduction-to-context?utm_source=chatgpt.com)

The actual execution flow:

```
                  USER
                   |
                   v
          Conversation Manager
                   |
                   v
            Token Manager
                   |
                   v
         Working Memory Manager
                   |
          +--------+--------+
          |        |        |
          v        v        v
        FIFO    Sliding   Priority
                Window    Memory
          |        |        |
          +--------+--------+
                   |
                   v
          Summarization Engine
                   |
                   v
             Context Builder
                   |
                   v
                  LLM
                   |
                   v
             Agent Response
                   |
                   v
           Update Conversation
                   |
                   v
           Persistent Storage
```

Note that the final design will also include a token-budget check after context construction. Summarization and selection will not be assumed to produce a valid context automatically.

# **6. Development milestones**

We'll build this in eight milestones. Each milestone has an independently testable outcome.

| Milestone | Implementation        | What you'll understand                |
| --------- | --------------------- | ------------------------------------- |
| M1        | Tokenization engine   | Tokens, encodings, budgets            |
| M2        | Conversation manager  | Message storage, roles, history       |
| M3        | Memory strategies     | Buffer, FIFO, sliding window          |
| M4        | Intelligent selection | Importance, recency, priority         |
| M5        | Summarization engine  | Full, rolling, incremental summaries  |
| M6        | Hybrid working memory | Summary + recent messages + key facts |
| M7        | Persistent memory     | SQLite, sessions, restoration         |
| M8        | DevMate application   | Complete coding assistant             |

Estimated effort: approximately 8–12 focused days, depending on experimentation and debugging.

# **7. Detailed milestone breakdown**

## **M1 — Tokenization engine**

**Goal:** Build a token-counting utility.

Implement:

- Text-to-token conversion.
- Token counting.
- Message-level token counting.
- Total conversation token counting.
- Input/output token budget separation.
- Context utilization calculation.

Test with:

- Short messages.
- Long messages.
- Source code.
- JSON.
- Multilingual text.

Deliverable: `token_manager.py`

## **M2 — Conversation manager**

**Goal:** Build your own conversation history manager.

Implement a message structure:

```
{    "id": "msg_001",    "role": "user",    "content": "I'm building an e-commerce application",    "timestamp": "...",    "token_count": 12}
```

Learn:

- Message roles.
- Message ordering.
- Conversation IDs.
- Session IDs.
- Message metadata.
- Tool messages.
- Conversation serialization.

Deliverable: `conversation_manager.py`

## **M3 — Memory strategies**

**Goal:** Implement different approaches to controlling conversation history.

Implement:

```
BufferMemory
FIFOMemory
SlidingWindowMemory
TokenBasedMemory
```

Experiment with different context limits and compare what information gets discarded.

Deliverable: `memory_strategies.py`

## **M4 — Intelligent memory selection**

**Goal:** Preserve important information instead of simply preserving recent information.

For example:

```
Message 1:
Our application uses PostgreSQL.

Message 2:
Hello!

Message 3:
How are you?

Message 4:
Our authentication system uses JWT.
```

If the system has room for only two messages, a naive sliding window might retain messages 3 and 4.

An intelligent memory selector should preserve messages 1 and 4.

Implement:

- Importance scoring.
- Recency scoring.
- Relevance scoring.
- Priority-based selection.
- Token-aware selection.

Deliverable: `memory_selector.py`

## **M5 — Summarization engine**

**Goal:** Implement conversation compression.

This is where we use an LLM.

Implement:

- Summarize complete conversations.
- Summarize older messages.
- Preserve recent messages.
- Update existing summaries.
- Maintain project-specific facts.
- Evaluate summary quality.

Deliverable: `summarization_engine.py`

## **M6 — Hybrid working memory**

**Goal:** Combine everything.

The context builder should construct:

```
SYSTEM PROMPT

PROJECT SUMMARY

IMPORTANT FACTS

RECENT CONVERSATION

CURRENT USER QUERY
```

Implement a context budget allocator, which reserves different portions of the context for each component.

Deliverable: `context_builder.py`

## **M7 — Persistent memory**

**Goal:** Make memory survive application restarts.

Implement:

- SQLite storage.
- Conversation persistence.
- Summary persistence.
- Message retrieval.
- Session restoration.
- Database migrations.

Deliverable: `storage.py`

## **M8 — Complete DevMate**

**Goal:** Integrate all components into a working AI coding assistant.

Features:

- Chat interface.
- Conversation history.
- Configurable token budget.
- Selectable memory strategies.
- Automatic summarization.
- Memory statistics.
- Context visualization.
- Persistent conversations.

Deliverable: A functional AI coding assistant.

# **8. Suggested folder structure**

Create the following project structure from the beginning. We'll populate it incrementally.

```
devmate/
│
├── app/
│   ├── main.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── token_manager.py
│   │
│   ├── conversation/
│   │   ├── models.py
│   │   └── conversation_manager.py
│   │
│   ├── memory/
│   │   ├── base.py
│   │   ├── buffer.py
│   │   ├── fifo.py
│   │   ├── sliding_window.py
│   │   ├── priority.py
│   │   └── hybrid.py
│   │
│   ├── summarization/
│   │   ├── summarizer.py
│   │   └── summary_manager.py
│   │
│   ├── context/
│   │   └── context_builder.py
│   │
│   └── storage/
│       ├── database.py
│       └── repository.py
│
├── tests/
│   ├── test_token_manager.py
│   ├── test_memory.py
│   └── test_summarization.py
│
├── data/
│
├── requirements.txt
├── .env.example
└── README.md
```

Don't create all these files and start implementing them at once. We'll create them as we reach each milestone.

# **9. How we'll evaluate our system**

We won't consider the project complete simply because the chatbot works.

We'll compare the strategies using the same conversation dataset.

| Metric               | What it measures                            |
| -------------------- | ------------------------------------------- |
| Token utilization    | How efficiently we use the context window   |
| Memory retention     | Whether important information survives      |
| Retrieval accuracy   | Whether relevant information is available   |
| Summary completeness | Whether important facts survive compression |
| Response accuracy    | Whether the agent answers correctly         |
| Latency              | Time required to construct context          |
| Cost                 | LLM token usage and summarization overhead  |

One particularly useful experiment:

Give DevMate 100 messages, place an important database decision in message 5, and ask about that decision at message 100.

Compare:

- Full conversation.
- FIFO.
- Sliding window.
- Summarization.
- Priority memory.
- Hybrid memory.

This will demonstrate why different memory strategies exist.

# **10. Your starting checklist**

Let's get the development environment ready.

**Environment setup**

## **0/7**

Create the devmate project directory

Create and activate a Python virtual environment

Install tiktoken and Pydantic

Create the initial folder structure

Create requirements.txt

Verify that the tokenizer works

Implement the first count_tokens() function

### **Initial setup commands**

For Windows PowerShell:

```
mkdir devmate
cd devmate

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install tiktoken pydantic
```

Create `requirements.txt`:

```
tiktoken
pydantic
```

### **First coding assignment**

Create `app/core/token_manager.py`.

Implement these functions:

```
def encode_text(text: str):    passdef count_tokens(text: str) -> int:    passdef count_message_tokens(messages: list) -> int:    passdef calculate_context_usage(    messages: list,    max_context_tokens: int,    reserved_output_tokens: int):    pass
```

For now, use `cl100k_base` as the tokenizer and clearly document that the production implementation must use the encoding appropriate to its target model.

**Don't implement the other functions using an LLM yet.** First build and test the tokenizer and budget calculations independently.

## **Definition of done for Project 1**

By the end, you should be able to explain and demonstrate:

- Why LLMs have context windows.
- How tokenization works.
- How context budgets are calculated.
- Why full conversation history is inefficient.
- How FIFO and sliding windows differ.
- Why summaries can lose information.
- How to preserve important facts.
- How to combine summaries and recent messages.
- How to persist and reconstruct working memory.

**Your immediate task:** Set up DevMate and finish M1. Share your `token_manager.py` implementation when ready. I'll review the code, test edge cases, explain improvements and then guide you into M2.