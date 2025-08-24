use async_openai::{config::OpenAIConfig, types::{ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs}, Client};
use horizons_core::{Environment, ToolCall};
use serde_json::json;
use std::env;
use tictactoe_env::{register_default_env, Config as TttConfig, TicTacToeEnvironment};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure model and API key
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5-nano".to_string());
    let api_key = env::var("OPENAI_API_KEY").expect("Set OPENAI_API_KEY to run the demo");
    let client = Client::with_config(OpenAIConfig::new().with_api_key(api_key));

    // Create environment with random opponent (minimax_prob=0.0)
    let mut env = TicTacToeEnvironment::new(TttConfig { opponent_minimax_prob: 0.0, ..Default::default() })?;
    let mut obs = env.initialize().await?;

    println!("Starting TicTacToe vs Random Opponent (model: {model})\n");
    loop {
        let data = &obs.data;
        let board = data["board_text"].as_str().unwrap();
        let to_move = data["to_move"].as_str().unwrap();
        let legal: Vec<usize> = data["legal_moves"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        println!("Board:\n{}\nTo move: {}\nLegal: {:?}\n", board, to_move, legal);

        if obs.terminated || obs.truncated { break; }

        // Ask the model to choose a move by index
        let prompt = format!(
            "You are X playing TicTacToe. Board is a 3x3 grid with indices 0..8 row-major. Legal moves: {:?}.\nReturn a JSON object {{\"index\": N}} with no extra text.",
            legal
        );
        let req = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages([
                ChatCompletionRequestMessageArgs::default().role(async_openai::types::Role::System).content("You are a helpful game-playing assistant.").build()?.into(),
                ChatCompletionRequestMessageArgs::default().role(async_openai::types::Role::User).content(prompt).build()?.into(),
            ])
            .build()?;
        let resp = client.chat().create(req).await?;
        let choice = resp.choices[0].message.content.clone().unwrap_or_default();
        let idx: usize = serde_json::from_str::<serde_json::Value>(&choice)
            .ok()
            .and_then(|v| v.get("index").and_then(|n| n.as_u64()))
            .map(|n| n as usize)
            .filter(|n| legal.contains(n))
            .unwrap_or_else(|| legal[0]);

        obs = env
            .step(vec![ToolCall { tool: "place".into(), args: json!({"index": idx}) }])
            .await?;
    }

    let winner = obs.data["winner"].clone();
    println!("Game over. Winner: {}", winner);
    Ok(())
}
