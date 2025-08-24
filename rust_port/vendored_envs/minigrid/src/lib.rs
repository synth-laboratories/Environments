pub mod types {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum Action {
        Left = 0,
        Right = 1,
        Forward = 2,
        Pickup = 3,
        Drop = 4,
        Toggle = 5,
        Done = 6,
    }

    impl TryFrom<u8> for Action {
        type Error = &'static str;
        fn try_from(v: u8) -> Result<Self, Self::Error> {
            Ok(match v {
                0 => Action::Left,
                1 => Action::Right,
                2 => Action::Forward,
                3 => Action::Pickup,
                4 => Action::Drop,
                5 => Action::Toggle,
                6 => Action::Done,
                _ => return Err("invalid action index (expected 0..6)"),
            })
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum Direction {
        Right = 0,
        Down = 1,
        Left = 2,
        Up = 3,
    }

    impl Direction {
        pub fn right(self) -> Self {
            match self {
                Direction::Right => Direction::Down,
                Direction::Down => Direction::Left,
                Direction::Left => Direction::Up,
                Direction::Up => Direction::Right,
            }
        }
        pub fn left(self) -> Self {
            match self {
                Direction::Right => Direction::Up,
                Direction::Up => Direction::Left,
                Direction::Left => Direction::Down,
                Direction::Down => Direction::Right,
            }
        }
        pub fn delta(self) -> (i32, i32) {
            match self {
                Direction::Right => (1, 0),
                Direction::Down => (0, 1),
                Direction::Left => (-1, 0),
                Direction::Up => (0, -1),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum ObjectKind {
        Unseen = 0,
        Empty = 1,
        Wall = 2,
        Door = 4,
        Key = 5,
        Ball = 6,
        Goal = 8,
        Lava = 9,
        Agent = 10,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct PublicState {
        pub grid_array: Vec<Vec<[u8; 3]>>, // [h][w][3] (object,color,state)
        pub agent_pos: (i32, i32),         // (x,y)
        pub agent_dir: u8,                 // 0..3
        pub step_count: u32,
        pub max_steps: u32,
        pub mission: String,
        pub terminated: bool,
        pub carrying: Option<(ObjectKind, u8)>, // (type, color)
    }
}

pub mod engine {
    use crate::types::{Action, Direction, ObjectKind, PublicState};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    enum Cell {
        Empty,
        Wall,
        Goal,
        Lava,
        Door { open: bool, locked: bool, color: u8 },
        Key { color: u8 },
        Ball { color: u8 },
    }

    #[derive(Debug, Clone)]
    pub struct MiniGridEnv {
        width: i32,
        height: i32,
        grid: Vec<Vec<Cell>>, // [y][x]
        agent_pos: (i32, i32),
        agent_dir: Direction,
        pub step_count: u32,
        pub max_steps: u32,
        pub terminated: bool,
        reward_last: f32,
        total_reward: f32,
        mission: String,
        carrying: Option<(ObjectKind, u8)>,
    }

    impl MiniGridEnv {
        pub fn empty_5x5() -> Self {
            let width = 5;
            let height = 5;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            for y in 0..height {
                for x in 0..width {
                    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                        grid[y as usize][x as usize] = Cell::Wall;
                    }
                }
            }
            grid[3][3] = Cell::Goal;
            Self {
                width,
                height,
                grid,
                agent_pos: (1, 1),
                agent_dir: Direction::Right,
                step_count: 0,
                max_steps: 100,
                terminated: false,
                reward_last: 0.0,
                total_reward: 0.0,
                mission: "Get to the green goal square".to_string(),
                carrying: None,
            }
        }

        pub fn doorkey_inline() -> Self {
            // 5x5: agent (1,1), key at (2,1) color 1, door at (3,1) locked, goal at (3,3)
            let width = 5;
            let height = 5;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            for y in 0..height {
                for x in 0..width {
                    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                        grid[y as usize][x as usize] = Cell::Wall;
                    }
                }
            }
            grid[1][2] = Cell::Key { color: 1 };
            grid[1][3] = Cell::Door { open: false, locked: true, color: 1 };
            grid[3][3] = Cell::Goal;
            Self {
                width,
                height,
                grid,
                agent_pos: (1, 1),
                agent_dir: Direction::Right,
                step_count: 0,
                max_steps: 100,
                terminated: false,
                reward_last: 0.0,
                total_reward: 0.0,
                mission: "Unlock the door and reach the goal".to_string(),
                carrying: None,
            }
        }

        pub fn four_rooms_19x19() -> Self {
            let width = 19;
            let height = 19;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            // Border walls
            for y in 0..height {
                for x in 0..width {
                    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                        grid[y as usize][x as usize] = Cell::Wall;
                    }
                }
            }
            // Vertical wall at center with two openings
            let midx = width / 2;
            for y in 1..(height - 1) {
                grid[y as usize][midx as usize] = Cell::Wall;
            }
            grid[5][midx as usize] = Cell::Empty;
            grid[13][midx as usize] = Cell::Empty;

            // Horizontal wall at center with two openings
            let midy = height / 2;
            for x in 1..(width - 1) {
                grid[midy as usize][x as usize] = Cell::Wall;
            }
            grid[midy as usize][5] = Cell::Empty;
            grid[midy as usize][13] = Cell::Empty;

            // Goal in bottom-right room
            grid[(height - 2) as usize][(width - 2) as usize] = Cell::Goal;

            Self {
                width,
                height,
                grid,
                agent_pos: (1, 1),
                agent_dir: Direction::Right,
                step_count: 0,
                max_steps: 400,
                terminated: false,
                reward_last: 0.0,
                total_reward: 0.0,
                mission: "Navigate through four rooms to reach the goal".to_string(),
                carrying: None,
            }
        }

        pub fn unlock_simple() -> Self {
            // 7x7 single locked door in a horizontal wall, key accessible, goal beyond
            let width = 7;
            let height = 7;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            for y in 0..height {
                for x in 0..width {
                    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                        grid[y as usize][x as usize] = Cell::Wall;
                    }
                }
            }
            // Horizontal wall across middle with a locked door at x=3
            for x in 1..(width - 1) {
                grid[(height / 2) as usize][x as usize] = Cell::Wall;
            }
            let dy = height / 2;
            grid[dy as usize][3] = Cell::Door { open: false, locked: true, color: 1 };
            // Key in upper room
            grid[1][2] = Cell::Key { color: 1 };
            // Goal in bottom room
            grid[(height - 2) as usize][(width - 2) as usize] = Cell::Goal;

            Self {
                width,
                height,
                grid,
                agent_pos: (1, 1),
                agent_dir: Direction::Right,
                step_count: 0,
                max_steps: 200,
                terminated: false,
                reward_last: 0.0,
                total_reward: 0.0,
                mission: "Unlock the door and reach the goal".to_string(),
                carrying: None,
            }
        }

        pub fn unlockpickup_simple() -> Self {
            // 7x7: locked door separates rooms; ball behind door; goal far corner
            let width = 7;
            let height = 7;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            for y in 0..height {
                for x in 0..width {
                    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                        grid[y as usize][x as usize] = Cell::Wall;
                    }
                }
            }
            // Vertical wall across middle with a locked door at y=3
            let midx = width / 2;
            for y in 1..(height - 1) {
                grid[y as usize][midx as usize] = Cell::Wall;
            }
            grid[3][midx as usize] = Cell::Door { open: false, locked: true, color: 2 };
            // Key near agent
            grid[1][2] = Cell::Key { color: 2 };
            // Ball behind door
            grid[4][(midx + 1) as usize] = Cell::Ball { color: 3 };
            // Goal in far corner
            grid[(height - 2) as usize][(width - 2) as usize] = Cell::Goal;

            Self {
                width,
                height,
                grid,
                agent_pos: (1, 1),
                agent_dir: Direction::Right,
                step_count: 0,
                max_steps: 200,
                terminated: false,
                reward_last: 0.0,
                total_reward: 0.0,
                mission: "Unlock, pick up the object, and reach the goal".to_string(),
                carrying: None,
            }
        }

        pub fn lava_inline() -> Self {
            // 5x5 with lava at (2,1)
            let width = 5;
            let height = 5;
            let mut grid = vec![vec![Cell::Empty; width as usize]; height as usize];
            for y in 0..height { for x in 0..width { if x == 0 || y == 0 || x == width - 1 || y == height - 1 { grid[y as usize][x as usize] = Cell::Wall; } } }
            grid[1][2] = Cell::Lava;
            grid[3][3] = Cell::Goal;
            Self { width, height, grid, agent_pos: (1,1), agent_dir: Direction::Right, step_count: 0, max_steps: 50, terminated: false, reward_last: 0.0, total_reward: 0.0, mission: "Avoid lava and reach the goal".into(), carrying: None }
        }

        fn in_bounds(&self, x: i32, y: i32) -> bool {
            x >= 0 && x < self.width && y >= 0 && y < self.height
        }

        fn passable(&self, x: i32, y: i32) -> bool {
            if !self.in_bounds(x, y) {
                return false;
            }
            match self.grid[y as usize][x as usize] {
                Cell::Empty => true,
                Cell::Goal => true,
                Cell::Lava => true, // entering lava is allowed but ends episode
                Cell::Door { open, .. } => open,
                Cell::Wall => false,
                Cell::Key { .. } => false, // cannot step onto key; must pickup first
                Cell::Ball { .. } => false,
            }
        }

        pub fn reset(&mut self) {
            *self = Self::empty_5x5();
        }

        fn to_array_cell(cell: &Cell, is_agent: bool, agent_dir: Direction) -> [u8; 3] {
            if is_agent {
                return [ObjectKind::Agent as u8, 0, agent_dir as u8];
            }
            match *cell {
                Cell::Empty => [ObjectKind::Empty as u8, 0, 0],
                Cell::Wall => [ObjectKind::Wall as u8, 0, 0],
                Cell::Goal => [ObjectKind::Goal as u8, 0, 0],
                Cell::Lava => [ObjectKind::Lava as u8, 0, 0],
                Cell::Door { open, locked, color } => {
                    // state channel: 0=closed, 1=open, 2=locked
                    let state = if open { 1 } else if locked { 2 } else { 0 };
                    [ObjectKind::Door as u8, color, state]
                }
                Cell::Key { color } => [ObjectKind::Key as u8, color, 0],
                Cell::Ball { color } => [ObjectKind::Ball as u8, color, 0],
            }
        }

        pub fn public_state(&self) -> PublicState {
            let mut grid = vec![vec![[ObjectKind::Empty as u8, 0, 0]; self.width as usize]; self.height as usize];
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_agent = self.agent_pos == (x, y);
                    if is_agent {
                        // For lava cells, preserve the lava but show agent direction
                        let cell = &self.grid[y as usize][x as usize];
                        match cell {
                            Cell::Lava => {
                                grid[y as usize][x as usize] = [ObjectKind::Lava as u8, 0, self.agent_dir as u8];
                            }
                            Cell::Key { color } => {
                                // Show the key with agent direction in state channel
                                grid[y as usize][x as usize] = [ObjectKind::Key as u8, *color, self.agent_dir as u8];
                            }
                            Cell::Ball { color } => {
                                // Show the ball with agent direction in state channel
                                grid[y as usize][x as usize] = [ObjectKind::Ball as u8, *color, self.agent_dir as u8];
                            }
                            _ => {
                                grid[y as usize][x as usize] = Self::to_array_cell(cell, true, self.agent_dir);
                            }
                        }
                    } else {
                        grid[y as usize][x as usize] = Self::to_array_cell(&self.grid[y as usize][x as usize], false, self.agent_dir);
                    }
                }
            }

            PublicState {
                grid_array: grid,
                agent_pos: self.agent_pos,
                agent_dir: self.agent_dir as u8,
                step_count: self.step_count,
                max_steps: self.max_steps,
                mission: self.mission.clone(),
                terminated: self.terminated,
                carrying: self.carrying,
            }
        }

        pub fn step(&mut self, action: Action) -> PublicState {
            if self.terminated {
                return self.public_state();
            }
            // Python MiniGrid increments step_count at the start of step()
            self.step_count += 1;
            // Apply step penalty like Python MiniGrid (-0.01 per step), but not for Done action
            let mut reward = if matches!(action, Action::Done) { 0.0 } else { -0.01 };

            match action {
                Action::Left => {
                    self.agent_dir = self.agent_dir.left();
                }
                Action::Right => {
                    self.agent_dir = self.agent_dir.right();
                }
                Action::Forward => {
                    let (dx, dy) = self.agent_dir.delta();
                    let nx = self.agent_pos.0 + dx;
                    let ny = self.agent_pos.1 + dy;
                    if self.passable(nx, ny) {
                        self.agent_pos = (nx, ny);
                        match self.grid[ny as usize][nx as usize] {
                            Cell::Lava => {
                                self.terminated = true;
                                // Lava gives no additional reward (just the step penalty)
                                reward = 0.0;
                            }
                            Cell::Goal => {
                                self.terminated = true;
                                // reward = 1 - 0.9 * (step_count / max_steps)
                                reward = 1.0 - 0.9 * (self.step_count as f32 / self.max_steps as f32);
                            }
                            _ => {}
                        }
                    }
                }
                Action::Pickup => {
                    let (dx, dy) = self.agent_dir.delta();
                    let nx = self.agent_pos.0 + dx;
                    let ny = self.agent_pos.1 + dy;
                    if self.in_bounds(nx, ny) && self.carrying.is_none() {
                        if let Cell::Key { color } = self.grid[ny as usize][nx as usize] {
                            // pick up key in front
                            self.grid[ny as usize][nx as usize] = Cell::Empty;
                            self.carrying = Some((ObjectKind::Key, color));
                        }
                    }
                }
                Action::Drop => {
                    // Drop in front of agent (symmetric with pickup)
                    let (dx, dy) = self.agent_dir.delta();
                    let nx = self.agent_pos.0 + dx;
                    let ny = self.agent_pos.1 + dy;
                    if self.in_bounds(nx, ny) {
                        if let Some((obj_type, color)) = self.carrying {
                            match obj_type {
                                ObjectKind::Key => {
                                    // Only drop if cell is empty (no object present)
                                    if let Cell::Empty = self.grid[ny as usize][nx as usize] {
                                        self.grid[ny as usize][nx as usize] = Cell::Key { color };
                                        self.carrying = None;
                                    }
                                }
                                ObjectKind::Ball => {
                                    // Only drop if cell is empty (no object present)
                                    if let Cell::Empty = self.grid[ny as usize][nx as usize] {
                                        self.grid[ny as usize][nx as usize] = Cell::Ball { color };
                                        self.carrying = None;
                                    }
                                }
                                _ => {} // Other object types not droppable
                            }
                        }
                    }
                }
                Action::Toggle => {
                    // Toggle door in front (open/close), unlock if carrying matching key
                    let (dx, dy) = self.agent_dir.delta();
                    let nx = self.agent_pos.0 + dx;
                    let ny = self.agent_pos.1 + dy;
                    if self.in_bounds(nx, ny) {
                        if let Cell::Door { open, locked, color } = self.grid[ny as usize][nx as usize] {
                            let mut new_open = open;
                            let mut new_locked = locked;
                            if locked {
                                if let Some((ObjectKind::Key, kcol)) = self.carrying {
                                    if kcol == color {
                                        new_locked = false;
                                        new_open = true;
                                    }
                                }
                            } else {
                                // toggle open/close
                                new_open = !open;
                            }
                            self.grid[ny as usize][nx as usize] = Cell::Door { open: new_open, locked: new_locked, color };
                        }
                    }
                }
                Action::Done => {
                    self.terminated = true;
                }
            }

            // Truncation on max steps
            if self.step_count >= self.max_steps {
                self.terminated = true;
            }
            self.reward_last = reward;
            self.total_reward += reward;
            self.public_state()
        }

        pub fn reward_last(&self) -> f32 { self.reward_last }
        pub fn total_reward(&self) -> f32 { self.total_reward }
        pub fn agent_dir_u8(&self) -> u8 { self.agent_dir as u8 }
        
        // Public accessors for private fields needed by tests
        pub fn width(&self) -> i32 { self.width }
        pub fn height(&self) -> i32 { self.height }
        pub fn mission(&self) -> &str { &self.mission }
    }
}
