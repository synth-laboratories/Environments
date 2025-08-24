use axum::{body::Body, http::{Request, StatusCode}};
use tower::ServiceExt; // for `oneshot`

#[tokio::test]
async fn tictactoe_minimax_block_via_http() {
    // Register env and build app
    tictactoe_env::register_default_env();
    let app = env_service::make_app();

    // GET /envs
    let res = app.clone().oneshot(Request::builder().uri("/envs").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = hyper::body::to_bytes(res.into_body()).await.unwrap();
    let arr: Vec<String> = serde_json::from_slice(&body).unwrap();
    assert!(arr.contains(&"TicTacToe".to_string()));

    // POST /initialize with minimax opponent
    let init_body = serde_json::json!({
        "env_type": "TicTacToe",
        "config": {"agent_mark":"X","opponent_minimax_prob":1.0,"seed":7}
    });
    let res = app.clone().oneshot(
        Request::builder()
            .method("POST")
            .uri("/initialize")
            .header("content-type","application/json")
            .body(Body::from(init_body.to_string()))
            .unwrap()
    ).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = hyper::body::to_bytes(res.into_body()).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let env_id = v.get("env_id").and_then(|x| x.as_str()).unwrap().to_string();

    // POST /step: X at idx 0
    let step1 = serde_json::json!({"env_id": env_id, "tool_calls": [{"tool":"place","args":{"index":0}}]});
    let res = app.clone().oneshot(
        Request::builder().method("POST").uri("/step").header("content-type","application/json").body(Body::from(step1.to_string())).unwrap()
    ).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // POST /step: X at idx 3, expect O blocks at 6
    let step2 = serde_json::json!({"env_id": v.get("env_id").unwrap().as_str().unwrap(), "tool_calls": [{"tool":"place","args":{"index":3}}]});
    let res = app.clone().oneshot(
        Request::builder().method("POST").uri("/step").header("content-type","application/json").body(Body::from(step2.to_string())).unwrap()
    ).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = hyper::body::to_bytes(res.into_body()).await.unwrap();
    let obs: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let legal: Vec<usize> = obs["data"]["legal_moves"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    assert!(!legal.contains(&6), "O should block at idx 6 under minimax");
}

