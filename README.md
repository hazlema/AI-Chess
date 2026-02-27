# ♔ Chess Arena

AI agents battle it out in this simple terminal version of chess.

No coaching. No hand-holding. Just a board, the rules, and whatever the models decide to do.

## What is this?

Chess Arena pits two AI models against each other via [OpenRouter](https://openrouter.ai). The game engine ([python-chess](https://python-chess.readthedocs.io/)) enforces all the rules — the models just have to pick legal moves. You'd be surprised how hard that is for some of them.

The system prompt is intentionally minimal. Models aren't told to analyze, strategize, or explain themselves. Some do anyway. That's the interesting part.

## What makes it fun

The real entertainment isn't the chess — it's the commentary. With zero prompting to "think out loud," models reveal wildly different personalities:

- **Grok** plays fast and cracks puns
- **Gemini** writes multi-paragraph essays comparing Sicilian variations with markdown formatting... then forfeits after three illegal moves
- **Sonnet** quietly analyzes endgame positions with surprising accuracy
- **GPT-4o** plays conservatively and keeps its thoughts to itself

Every game saves a full commentary log so you can read exactly what each model was thinking (or pretending to think) on every turn.

## Setup

```bash
pip install python-chess requests
export OPENROUTER_API_KEY=your_key_here
```

## Usage

```bash
# Defaults: claude-sonnet-4 vs gpt-4o
python chess_arena.py

# Pick your fighters
python chess_arena.py --white x-ai/grok-3 --black google/gemini-2.5-flash

# Slow it down to read the commentary live
python chess_arena.py --white x-ai/grok-3 --black google/gemini-2.5-pro --delay 2
```

Use any model available on [OpenRouter](https://openrouter.ai/models).

## Output

Each game produces:

| File | What's in it |
|------|-------------|
| `chess_*_.pgn` | Standard PGN with AI commentary as annotations — loadable in Lichess, chess.com, or any analysis tool |
| `chess_*_commentary.txt` | Full untruncated model responses for every move |

## Features

- Board with color, last-move highlighting, and captured pieces display
- 3 retries per move if a model returns an illegal move
- Auto-detects checkmate, stalemate, draws, and forfeits
- PGN export with embedded commentary
- Minimal system prompt by design — see what the models actually do when no one's directing them

## Fun matchups to try

| White | Black | Vibe |
|-------|-------|------|
| `x-ai/grok-3` | `google/gemini-2.5-pro` | Class clown vs. the professor |
| `anthropic/claude-sonnet-4` | `openai/gpt-4o` | Quiet strategists |
| `meta-llama/llama-4-maverick` | `anthropic/claude-sonnet-4` | Wildcard energy |
| `openai/gpt-4o` | `x-ai/grok-3` | Buttoned-up vs. unhinged |

## Why minimal prompting?

Most AI chess projects over-engineer the prompt to make models play well. This one is designed to make them play *honestly*. The system prompt is basically "you're playing chess, here's the board, your move." What they choose to say beyond that is entirely on them.

This project was born out of [AgentArcade](https://agentarcade.co) research into AI behavioral evaluation — testing what models *actually do* vs. what they claim they'll do.

## License

MIT
