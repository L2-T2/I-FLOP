use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;

#[derive(Debug)]
struct NativeError(String);

impl fmt::Display for NativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for NativeError {}

type NativeResult<T> = Result<T, NativeError>;

#[derive(Clone)]
struct EnvData {
    id: i32,
    n: usize,
    p: usize,
    values: Vec<f64>,
    targets: Vec<usize>,
}

#[derive(Clone)]
struct Dataset {
    p: usize,
    envs: Vec<EnvData>,
}

impl Dataset {
    fn total_samples(&self) -> usize {
        self.envs.iter().map(|env| env.n).sum()
    }

    fn effective_env_indices(&self, node: usize) -> Vec<usize> {
        self.envs
            .iter()
            .enumerate()
            .filter_map(|(idx, env)| {
                if env.targets.contains(&node) {
                    None
                } else {
                    Some(idx)
                }
            })
            .collect()
    }

    fn observational_env_indices(&self) -> Vec<usize> {
        self.envs
            .iter()
            .enumerate()
            .filter_map(|(idx, env)| if env.targets.is_empty() { Some(idx) } else { None })
            .collect()
    }

    fn stacked_rows(&self, indices: &[usize]) -> Matrix {
        let n: usize = indices.iter().map(|idx| self.envs[*idx].n).sum();
        let mut out = vec![0.0; n * self.p];
        let mut dest = 0;
        for idx in indices {
            let env = &self.envs[*idx];
            for row in 0..env.n {
                for col in 0..self.p {
                    out[dest * self.p + col] = env.values[row * self.p + col];
                }
                dest += 1;
            }
        }
        Matrix {
            n,
            p: self.p,
            values: out,
        }
    }

}

#[derive(Clone)]
struct Matrix {
    n: usize,
    p: usize,
    values: Vec<f64>,
}

impl Matrix {
    fn at(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.p + col]
    }
}

#[derive(Clone)]
struct SearchConfig {
    ils_restarts: usize,
    perturbation_size: Option<usize>,
    random_seed: u64,
    max_sweeps: Option<usize>,
    atol: f64,
}

#[derive(Clone)]
struct GiesConfig {
    eps: f64,
    penalty_sample_mode: String,
    fit_weight: f64,
    penalty_weight: f64,
    envwise_residual_mode: String,
}

#[derive(Clone)]
struct Request {
    mode: String,
    score_key: String,
    dataset: Dataset,
    search: SearchConfig,
    gies: GiesConfig,
    order: Option<Vec<usize>>,
}

#[derive(Clone)]
struct Candidate {
    order: Vec<usize>,
    score: f64,
    parents: Vec<Vec<usize>>,
    dag_adjacency: Vec<Vec<u8>>,
    adjacency: Vec<Vec<u8>>,
    adjacency_type: String,
    score_vector: Option<(usize, usize)>,
}

#[derive(Clone)]
struct GiesNodeState {
    node: usize,
    parents: Vec<usize>,
    score: f64,
    env_states: Vec<EnvLocalState>,
}

#[derive(Clone)]
struct EnvLocalState {
    env_index: usize,
    n: usize,
    sigma2: f64,
    chol: Option<Cholesky>,
}

#[derive(Clone)]
struct GiesSearchState {
    order: Vec<usize>,
    local_scores: Vec<GiesNodeState>,
    total_score: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalBicScoreMode {
    GiesEnvwiseTargetFiltered,
    FlopObs,
    FlopEnvwise,
}

struct GiesScorer<'a> {
    dataset: &'a Dataset,
    eps: f64,
    penalty_sample_mode: String,
    fit_weight: f64,
    penalty_weight: f64,
    envwise_residual_mode: String,
    mode: LocalBicScoreMode,
    env_scatters: Vec<Vec<f64>>,
    obs_scatter: Option<(usize, Vec<f64>)>,
    pooled_scatter_by_node: Vec<Option<(usize, Vec<f64>)>>,
    cache: HashMap<String, f64>,
}

impl<'a> GiesScorer<'a> {
    fn new(dataset: &'a Dataset, config: &GiesConfig) -> Self {
        let env_scatters = precompute_env_scatters(dataset);
        let pooled_scatter_by_node = precompute_pooled_scatters_by_node(
            dataset,
            &env_scatters,
            LocalBicScoreMode::GiesEnvwiseTargetFiltered,
        );
        Self {
            dataset,
            eps: config.eps,
            penalty_sample_mode: config.penalty_sample_mode.clone(),
            fit_weight: config.fit_weight,
            penalty_weight: config.penalty_weight,
            envwise_residual_mode: config.envwise_residual_mode.clone(),
            mode: LocalBicScoreMode::GiesEnvwiseTargetFiltered,
            env_scatters,
            obs_scatter: None,
            pooled_scatter_by_node,
            cache: HashMap::new(),
        }
    }

    fn new_flop_obs(dataset: &'a Dataset, config: &GiesConfig) -> Self {
        let obs_scatter = precompute_observational_scatter(dataset);
        Self {
            dataset,
            eps: config.eps,
            penalty_sample_mode: "effective".to_string(),
            fit_weight: config.fit_weight,
            penalty_weight: config.penalty_weight,
            envwise_residual_mode: config.envwise_residual_mode.clone(),
            mode: LocalBicScoreMode::FlopObs,
            env_scatters: precompute_env_scatters(dataset),
            obs_scatter,
            pooled_scatter_by_node: Vec::new(),
            cache: HashMap::new(),
        }
    }

    fn new_flop_envwise(dataset: &'a Dataset, config: &GiesConfig) -> Self {
        let env_scatters = precompute_env_scatters(dataset);
        let pooled_scatter_by_node = precompute_pooled_scatters_by_node(
            dataset,
            &env_scatters,
            LocalBicScoreMode::FlopEnvwise,
        );
        Self {
            dataset,
            eps: config.eps,
            penalty_sample_mode: config.penalty_sample_mode.clone(),
            fit_weight: config.fit_weight,
            penalty_weight: config.penalty_weight,
            envwise_residual_mode: config.envwise_residual_mode.clone(),
            mode: LocalBicScoreMode::FlopEnvwise,
            env_scatters,
            obs_scatter: None,
            pooled_scatter_by_node,
            cache: HashMap::new(),
        }
    }

    fn local_score(&mut self, node: usize, parents: &[usize]) -> NativeResult<f64> {
        let pset = sorted_unique(parents);
        let key = format!("{}:{}", node, join_usize(&pset));
        if let Some(value) = self.cache.get(&key) {
            return Ok(*value);
        }
        let value = self.local_score_uncached(node, &pset)?;
        self.cache.insert(key, value);
        Ok(value)
    }

    fn local_state(&mut self, node: usize, parents: &[usize]) -> NativeResult<GiesNodeState> {
        let pset = sorted_unique(parents);
        let state = self.local_state_uncached(node, &pset)?;
        let key = format!("{}:{}", node, join_usize(&pset));
        self.cache.insert(key, state.score);
        Ok(state)
    }

    fn local_score_uncached(&self, node: usize, parents: &[usize]) -> NativeResult<f64> {
        Ok(self.local_state_uncached(node, parents)?.score)
    }

    fn local_state_uncached(&self, node: usize, parents: &[usize]) -> NativeResult<GiesNodeState> {
        if self.mode == LocalBicScoreMode::FlopObs {
            return self.local_state_flop_obs(node, parents);
        }
        let env_indices = if self.mode == LocalBicScoreMode::FlopEnvwise {
            (0..self.dataset.envs.len()).collect::<Vec<_>>()
        } else {
            self.dataset.effective_env_indices(node)
        };
        if env_indices.is_empty() {
            return Err(NativeError(format!("node {} has no effective environments", node)));
        }
        if self.mode == LocalBicScoreMode::GiesEnvwiseTargetFiltered
            || (self.mode == LocalBicScoreMode::FlopEnvwise
                && self.envwise_residual_mode == "pooled_covariance")
        {
            return self.local_state_iflop_envwise_shared(node, parents, &env_indices);
        }
        let mut n_fit = 0usize;
        let mut env_states = Vec::<EnvLocalState>::with_capacity(env_indices.len());
        for idx in &env_indices {
            let env = &self.dataset.envs[*idx];
            env_states.push(self.env_local_state_from_scratch(*idx, node, parents)?);
            n_fit += env.n;
        }
        let n_penalty = self.penalty_n(node, &env_indices, n_fit)?;
        let score = self.score_from_env_states(&env_states, n_penalty, parents.len());
        Ok(GiesNodeState {
            node,
            parents: parents.to_vec(),
            score,
            env_states,
        })
    }

    fn local_state_flop_obs(&self, node: usize, parents: &[usize]) -> NativeResult<GiesNodeState> {
        let (n, scatter) = self.obs_scatter.as_ref().ok_or_else(|| {
            NativeError("flop_obs requires at least one observational environment".to_string())
        })?;
        let env_state = self.local_state_for_scatter(usize::MAX, *n, scatter, node, parents)?;
        let score = self.score_from_env_states(std::slice::from_ref(&env_state), *n, parents.len());
        Ok(GiesNodeState {
            node,
            parents: parents.to_vec(),
            score,
            env_states: vec![env_state],
        })
    }

    fn local_state_plus(
        &mut self,
        local: &GiesNodeState,
        parent: usize,
    ) -> NativeResult<GiesNodeState> {
        if local.parents.contains(&parent) {
            return Ok(local.clone());
        }
        let mut parents = local.parents.clone();
        parents.push(parent);
        if self.mode == LocalBicScoreMode::GiesEnvwiseTargetFiltered
            && self.envwise_residual_mode != "pooled_covariance"
        {
            return self.local_state(local.node, &parents);
        }
        let mut env_states = Vec::<EnvLocalState>::with_capacity(local.env_states.len());
        for env_state in &local.env_states {
            env_states.push(self.env_state_plus(local.node, &local.parents, env_state, parent)?);
        }
        let n_fit: usize = env_states.iter().map(|state| state.n).sum();
        let env_indices = concrete_env_indices(&env_states);
        let n_penalty = if self.mode == LocalBicScoreMode::FlopObs {
            n_fit
        } else if env_indices.is_empty() {
            self.penalty_n_for_pooled_state(local.node, n_fit)?
        } else {
            self.penalty_n(local.node, &env_indices, n_fit)?
        };
        let score = self.score_from_env_states(&env_states, n_penalty, parents.len());
        let state = GiesNodeState {
            node: local.node,
            parents,
            score,
            env_states,
        };
        let key = format!("{}:{}", state.node, join_usize(&state.parents));
        self.cache.insert(key, state.score);
        Ok(state)
    }

    fn local_state_minus(
        &mut self,
        local: &GiesNodeState,
        parent: usize,
    ) -> NativeResult<GiesNodeState> {
        if !local.parents.contains(&parent) {
            return Ok(local.clone());
        }
        let parents: Vec<usize> = local
            .parents
            .iter()
            .copied()
            .filter(|item| *item != parent)
            .collect();
        if self.mode == LocalBicScoreMode::GiesEnvwiseTargetFiltered
            && self.envwise_residual_mode != "pooled_covariance"
        {
            return self.local_state(local.node, &parents);
        }
        let mut env_states = Vec::<EnvLocalState>::with_capacity(local.env_states.len());
        for env_state in &local.env_states {
            env_states.push(self.env_state_minus(local.node, &local.parents, env_state, parent)?);
        }
        let n_fit: usize = env_states.iter().map(|state| state.n).sum();
        let env_indices = concrete_env_indices(&env_states);
        let n_penalty = if self.mode == LocalBicScoreMode::FlopObs {
            n_fit
        } else if env_indices.is_empty() {
            self.penalty_n_for_pooled_state(local.node, n_fit)?
        } else {
            self.penalty_n(local.node, &env_indices, n_fit)?
        };
        let score = self.score_from_env_states(&env_states, n_penalty, parents.len());
        let state = GiesNodeState {
            node: local.node,
            parents,
            score,
            env_states,
        };
        let key = format!("{}:{}", state.node, join_usize(&state.parents));
        self.cache.insert(key, state.score);
        Ok(state)
    }

    fn total_score(&mut self, parents: &[Vec<usize>]) -> NativeResult<f64> {
        let mut total = 0.0;
        for node in 0..self.dataset.p {
            total += self.local_score(node, &parents[node])?;
        }
        Ok(total)
    }

    fn local_state_iflop_envwise_shared(
        &self,
        node: usize,
        parents: &[usize],
        env_indices: &[usize],
    ) -> NativeResult<GiesNodeState> {
        if self.envwise_residual_mode == "pooled_covariance" {
            let (n_fit, pooled_scatter) = self
                .pooled_scatter_by_node
                .get(node)
                .and_then(|item| item.as_ref())
                .ok_or_else(|| NativeError(format!("missing pooled scatter for node {}", node)))?;
            let chol = cholesky_for_node_parents(pooled_scatter, self.dataset.p, node, parents);
            let sigma2 = if let Some(chol_value) = chol.as_ref() {
                let stddev_res = chol_value.get_bottom_right();
                ((stddev_res * stddev_res) / ((*n_fit).max(1) as f64)).max(self.eps)
            } else {
                residual_variance_scatter(
                    pooled_scatter,
                    *n_fit,
                    self.dataset.p,
                    node,
                    parents,
                    self.eps,
                )
            };
            let env_state = EnvLocalState {
                env_index: usize::MAX,
                n: *n_fit,
                sigma2,
                chol,
            };
            let n_penalty = self.penalty_n(node, env_indices, *n_fit)?;
            let score = self.score_from_env_states(std::slice::from_ref(&env_state), n_penalty, parents.len());
            return Ok(GiesNodeState {
                node,
                parents: parents.to_vec(),
                score,
                env_states: vec![env_state],
            });
        }
        let mut pooled_scatter = vec![0.0; self.dataset.p * self.dataset.p];
        let mut n_fit = 0usize;
        for idx in env_indices {
            let env = &self.dataset.envs[*idx];
            n_fit += env.n;
            let scatter = &self.env_scatters[*idx];
            for pos in 0..pooled_scatter.len() {
                pooled_scatter[pos] += scatter[pos];
            }
        }
        if self.envwise_residual_mode != "env_residuals" {
            return Err(NativeError(format!(
                "unknown envwise residual mode {}",
                self.envwise_residual_mode
            )));
        }
        let beta = shared_beta_from_scatter(
            &pooled_scatter,
            self.dataset.p,
            node,
            parents,
            self.eps,
        );
        let mut env_states = Vec::<EnvLocalState>::with_capacity(env_indices.len());
        for idx in env_indices {
            let env = &self.dataset.envs[*idx];
            let scatter = &self.env_scatters[*idx];
            let sigma2 = residual_variance_with_beta(
                scatter,
                env.n,
                self.dataset.p,
                node,
                parents,
                &beta,
                self.eps,
            );
            env_states.push(EnvLocalState {
                env_index: *idx,
                n: env.n,
                sigma2,
                chol: None,
            });
        }
        let n_penalty = self.penalty_n(node, env_indices, n_fit)?;
        let score = self.score_from_env_states(&env_states, n_penalty, parents.len());
        Ok(GiesNodeState {
            node,
            parents: parents.to_vec(),
            score,
            env_states,
        })
    }

    fn env_local_state_from_scratch(
        &self,
        env_index: usize,
        node: usize,
        parents: &[usize],
    ) -> NativeResult<EnvLocalState> {
        let env = &self.dataset.envs[env_index];
        self.local_state_for_scatter(env_index, env.n, &self.env_scatters[env_index], node, parents)
    }

    fn local_state_for_scatter(
        &self,
        env_index: usize,
        n: usize,
        scatter: &[f64],
        node: usize,
        parents: &[usize],
    ) -> NativeResult<EnvLocalState> {
        let chol = cholesky_for_node_parents(scatter, self.dataset.p, node, parents);
        let sigma2 = if let Some(chol_value) = chol.as_ref() {
            let stddev_res = chol_value.get_bottom_right();
            ((stddev_res * stddev_res) / (n.max(1) as f64)).max(self.eps)
        } else {
            residual_variance_scatter(scatter, n, self.dataset.p, node, parents, self.eps)
        };
        Ok(EnvLocalState {
            env_index,
            n,
            sigma2,
            chol,
        })
    }

    fn env_state_plus(
        &self,
        node: usize,
        old_parents: &[usize],
        old_state: &EnvLocalState,
        parent: usize,
    ) -> NativeResult<EnvLocalState> {
        let (n, scatter) = self.scatter_for_env_state(old_state, node)?;
        if let Some(chol) = old_state.chol.as_ref() {
            let mut items = old_parents.to_vec();
            items.push(node);
            items.push(parent);
            let ins_col = column_subvector_flat(scatter, self.dataset.p, &items, parent);
            if let Some(new_chol) = chol.insert_column_before_last(ins_col) {
                let stddev_res = new_chol.get_bottom_right();
                return Ok(EnvLocalState {
                    env_index: old_state.env_index,
                    n,
                    sigma2: ((stddev_res * stddev_res) / (n.max(1) as f64)).max(self.eps),
                    chol: Some(new_chol),
                });
            }
        }
        let mut parents = old_parents.to_vec();
        parents.push(parent);
        self.local_state_for_scatter(old_state.env_index, n, scatter, node, &parents)
    }

    fn env_state_minus(
        &self,
        node: usize,
        old_parents: &[usize],
        old_state: &EnvLocalState,
        parent: usize,
    ) -> NativeResult<EnvLocalState> {
        let (n, scatter) = self.scatter_for_env_state(old_state, node)?;
        if let Some(chol) = old_state.chol.as_ref() {
            if let Some(idx) = old_parents.iter().position(|item| *item == parent) {
                let new_chol = chol.remove_column(idx);
                let stddev_res = new_chol.get_bottom_right();
                return Ok(EnvLocalState {
                    env_index: old_state.env_index,
                    n,
                    sigma2: ((stddev_res * stddev_res) / (n.max(1) as f64)).max(self.eps),
                    chol: Some(new_chol),
                });
            }
        }
        let parents: Vec<usize> = old_parents.iter().copied().filter(|item| *item != parent).collect();
        self.local_state_for_scatter(old_state.env_index, n, scatter, node, &parents)
    }

    fn scatter_for_env_state<'b>(&'b self, state: &EnvLocalState, node: usize) -> NativeResult<(usize, &'b [f64])> {
        if state.env_index == usize::MAX {
            if self.mode != LocalBicScoreMode::FlopObs {
                let (n, scatter) = self
                    .pooled_scatter_by_node
                    .get(node)
                    .and_then(|item| item.as_ref())
                    .ok_or_else(|| NativeError(format!("missing pooled scatter for node {}", node)))?;
                return Ok((*n, scatter.as_slice()));
            }
            let (n, scatter) = self.obs_scatter.as_ref().ok_or_else(|| {
                NativeError("missing precomputed observational scatter".to_string())
            })?;
            Ok((*n, scatter.as_slice()))
        } else {
            Ok((
                self.dataset.envs[state.env_index].n,
                self.env_scatters[state.env_index].as_slice(),
            ))
        }
    }

    fn score_from_env_states(&self, env_states: &[EnvLocalState], n_penalty: usize, num_parents: usize) -> f64 {
        let fit: f64 = env_states
            .iter()
            .map(|state| 0.5 * (state.n as f64) * (1.0 + state.sigma2.ln()))
            .sum();
        let penalty = 0.5 * ((n_penalty.max(2)) as f64).ln() * ((num_parents + 1) as f64);
        self.fit_weight * fit + self.penalty_weight * penalty
    }

    fn penalty_n(&self, node: usize, env_indices: &[usize], n_fit: usize) -> NativeResult<usize> {
        match self.penalty_sample_mode.as_str() {
            "total" => Ok(self.dataset.total_samples()),
            "effective" => Ok(n_fit),
            "max_env" => Ok(env_indices
                .iter()
                .map(|idx| self.dataset.envs[*idx].n)
                .max()
                .unwrap_or(1)),
            other => Err(NativeError(format!(
                "unknown penalty_sample_mode {} for node {}",
                other, node
            ))),
        }
    }

    fn penalty_n_for_pooled_state(&self, node: usize, n_fit: usize) -> NativeResult<usize> {
        match self.penalty_sample_mode.as_str() {
            "total" => Ok(self.dataset.total_samples()),
            "effective" | "max_env" => Ok(n_fit),
            other => Err(NativeError(format!(
                "unknown penalty_sample_mode {} for pooled node {}",
                other, node
            ))),
        }
    }
}

fn main() {
    if let Err(err) = run_main() {
        println!("IFLOP_NATIVE_RESULT_V1");
        println!("status error");
        println!("failure_reason {}", sanitize_line(&err.to_string()));
        println!("END");
        std::process::exit(1);
    }
}

fn run_main() -> NativeResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(NativeError(
            "usage: iflop_native <request-file>".to_string(),
        ));
    }
    let text = fs::read_to_string(&args[1]).map_err(|err| NativeError(err.to_string()))?;
    let request = parse_request(&text)?;
    let mut result = if request.mode == "eval_order" {
        let order = request
            .order
            .clone()
            .ok_or_else(|| NativeError("eval_order mode requires ORDER".to_string()))?;
        evaluate_order_request(&request, &order)?
    } else if request.mode == "run" {
        run_search_request(&request)?
    } else {
        return Err(NativeError(format!("unknown mode {}", request.mode)));
    };
    apply_output_semantics(&request, &mut result);
    print_candidate(&request.score_key, &result);
    Ok(())
}

fn apply_output_semantics(request: &Request, candidate: &mut Candidate) {
    if request.score_key == "i_flop_envwise" {
        candidate.adjacency = icpdag_adjacency(&candidate.dag_adjacency, &request.dataset);
        candidate.adjacency_type = "i_cpdag".to_string();
    }
}

fn evaluate_order_request(request: &Request, order: &[usize]) -> NativeResult<Candidate> {
    validate_order(order, request.dataset.p)?;
    match request.score_key.as_str() {
        "i_flop_envwise" => {
            let mut scorer = GiesScorer::new(&request.dataset, &request.gies);
            let parents = parent_sets_for_order_gies(&mut scorer, order, request.search.atol)?;
            let dag_adjacency = adjacency_from_parents(&parents, request.dataset.p);
            let adjacency = cpdag_adjacency_from_parents(&parents, request.dataset.p);
            let score = scorer.total_score(&parents)?;
            Ok(Candidate {
                order: order.to_vec(),
                score,
                parents,
                dag_adjacency,
                adjacency,
                adjacency_type: "cpdag".to_string(),
                score_vector: None,
            })
        }
        "flop_obs" => {
            let mut scorer = GiesScorer::new_flop_obs(&request.dataset, &request.gies);
            let parents = parent_sets_for_order_gies(&mut scorer, order, request.search.atol)?;
            let dag_adjacency = adjacency_from_parents(&parents, request.dataset.p);
            let adjacency = cpdag_adjacency_from_parents(&parents, request.dataset.p);
            let score = scorer.total_score(&parents)?;
            Ok(Candidate {
                order: order.to_vec(),
                score,
                parents,
                dag_adjacency,
                adjacency,
                adjacency_type: "cpdag".to_string(),
                score_vector: None,
            })
        }
        "flop_envwise" => {
            let mut scorer = GiesScorer::new_flop_envwise(&request.dataset, &request.gies);
            let parents = parent_sets_for_order_gies(&mut scorer, order, request.search.atol)?;
            let dag_adjacency = adjacency_from_parents(&parents, request.dataset.p);
            let adjacency = cpdag_adjacency_from_parents(&parents, request.dataset.p);
            let score = scorer.total_score(&parents)?;
            Ok(Candidate {
                order: order.to_vec(),
                score,
                parents,
                dag_adjacency,
                adjacency,
                adjacency_type: "cpdag".to_string(),
                score_vector: None,
            })
        }
        other => Err(NativeError(format!(
            "native backend currently supports flop_obs, flop_envwise and i_flop_envwise, got {}",
            other
        ))),
    }
}

fn run_search_request(request: &Request) -> NativeResult<Candidate> {
    match request.score_key.as_str() {
        "i_flop_envwise" => {
            let mut scorer = GiesScorer::new(&request.dataset, &request.gies);
            run_iflop_order_search(&mut scorer, &request.search)
        }
        "flop_obs" => {
            let mut scorer = GiesScorer::new_flop_obs(&request.dataset, &request.gies);
            run_iflop_order_search(&mut scorer, &request.search)
        }
        "flop_envwise" => {
            let mut scorer = GiesScorer::new_flop_envwise(&request.dataset, &request.gies);
            run_iflop_order_search(&mut scorer, &request.search)
        }
        other => Err(NativeError(format!(
            "native backend currently supports flop_obs, flop_envwise and i_flop_envwise, got {}",
            other
        ))),
    }
}

fn parse_request(text: &str) -> NativeResult<Request> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| NativeError("empty native request".to_string()))?
        .trim();
    if header != "IFLOP_NATIVE_V1" {
        return Err(NativeError(format!("bad request header {}", header)));
    }

    let mut mode = "run".to_string();
    let mut score_key = "i_flop_envwise".to_string();
    let mut p: Option<usize> = None;
    let mut envs: Vec<EnvData> = Vec::new();
    let mut search = SearchConfig {
        ils_restarts: 2,
        perturbation_size: None,
        random_seed: 0,
        max_sweeps: None,
        atol: 1.0e-10,
    };
    let mut gies = GiesConfig {
        eps: 1.0e-8,
        penalty_sample_mode: "total".to_string(),
        fit_weight: 1.0,
        penalty_weight: 1.0,
        envwise_residual_mode: "pooled_covariance".to_string(),
    };
    let mut order: Option<Vec<usize>> = None;

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "MODE" => mode = expect_value(&parts, "MODE")?.to_string(),
            "SCORE_KEY" => score_key = expect_value(&parts, "SCORE_KEY")?.to_string(),
            "P" => p = Some(parse_usize(expect_value(&parts, "P")?, "P")?),
            "EPS" => gies.eps = parse_f64(expect_value(&parts, "EPS")?, "EPS")?,
            "PENALTY_SAMPLE_MODE" => {
                gies.penalty_sample_mode = expect_value(&parts, "PENALTY_SAMPLE_MODE")?.to_string()
            }
            "GIES_FIT_WEIGHT" => {
                gies.fit_weight = parse_f64(expect_value(&parts, "GIES_FIT_WEIGHT")?, "GIES_FIT_WEIGHT")?
            }
            "GIES_PENALTY_WEIGHT" => {
                gies.penalty_weight =
                    parse_f64(expect_value(&parts, "GIES_PENALTY_WEIGHT")?, "GIES_PENALTY_WEIGHT")?
            }
            "GIES_ENVWISE_RESIDUAL_MODE" => {
                gies.envwise_residual_mode = expect_value(&parts, "GIES_ENVWISE_RESIDUAL_MODE")?.to_string()
            }
            "SEARCH_ILS" => {
                let value = parse_isize(expect_value(&parts, "SEARCH_ILS")?, "SEARCH_ILS")?;
                search.ils_restarts = value.max(0) as usize;
            }
            "RANDOM_SEED" => {
                search.random_seed = parse_u64(expect_value(&parts, "RANDOM_SEED")?, "RANDOM_SEED")?
            }
            "MAX_SWEEPS" => {
                let value = expect_value(&parts, "MAX_SWEEPS")?;
                search.max_sweeps = if value == "none" {
                    None
                } else {
                    Some(parse_usize(value, "MAX_SWEEPS")?)
                };
            }
            "PERTURBATION_SIZE" => {
                let value = expect_value(&parts, "PERTURBATION_SIZE")?;
                search.perturbation_size = if value == "none" {
                    None
                } else {
                    Some(parse_usize(value, "PERTURBATION_SIZE")?)
                };
            }
            "ATOL" => search.atol = parse_f64(expect_value(&parts, "ATOL")?, "ATOL")?,
            "ORDER" => {
                if parts.len() < 2 {
                    return Err(NativeError("ORDER requires length".to_string()));
                }
                let len = parse_usize(parts[1], "ORDER length")?;
                if parts.len() != len + 2 {
                    return Err(NativeError("ORDER length does not match values".to_string()));
                }
                let mut values = Vec::with_capacity(len);
                for item in &parts[2..] {
                    values.push(parse_usize(item, "ORDER value")?);
                }
                order = Some(values);
            }
            "ENV" => {
                if parts.len() < 4 {
                    return Err(NativeError("ENV requires id n target_count".to_string()));
                }
                let env_id = parse_i32(parts[1], "ENV id")?;
                let n = parse_usize(parts[2], "ENV n")?;
                let target_count = parse_usize(parts[3], "ENV target_count")?;
                if parts.len() != target_count + 4 {
                    return Err(NativeError("ENV target_count does not match values".to_string()));
                }
                let p_value = p.ok_or_else(|| NativeError("P must appear before ENV".to_string()))?;
                let mut targets = Vec::<usize>::with_capacity(target_count);
                for item in &parts[4..] {
                    targets.push(parse_usize(item, "ENV target")?);
                }
                targets.sort_unstable();
                targets.dedup();
                let mut values = Vec::<f64>::with_capacity(n * p_value);
                for row_idx in 0..n {
                    let row_line = lines.next().ok_or_else(|| {
                        NativeError(format!("ENV {} missing row {}", env_id, row_idx))
                    })?;
                    let row_parts: Vec<&str> = row_line.split_whitespace().collect();
                    if row_parts.len() != p_value {
                        return Err(NativeError(format!(
                            "ENV {} row {} expected {} values, got {}",
                            env_id,
                            row_idx,
                            p_value,
                            row_parts.len()
                        )));
                    }
                    for item in row_parts {
                        values.push(parse_f64(item, "ENV matrix value")?);
                    }
                }
                envs.push(EnvData {
                    id: env_id,
                    n,
                    p: p_value,
                    values,
                    targets,
                });
            }
            "END" => break,
            other => return Err(NativeError(format!("unknown request field {}", other))),
        }
    }

    let p_value = p.ok_or_else(|| NativeError("P is required".to_string()))?;
    if envs.is_empty() {
        return Err(NativeError("at least one ENV is required".to_string()));
    }
    for env in &envs {
        if env.p != p_value {
            return Err(NativeError(format!("ENV {} has inconsistent P", env.id)));
        }
        for target in &env.targets {
            if *target >= p_value {
                return Err(NativeError(format!(
                    "ENV {} has invalid intervention target {}",
                    env.id, target
                )));
            }
        }
    }
    envs.sort_by_key(|env| env.id);
    let dataset = Dataset { p: p_value, envs };
    Ok(Request {
        mode,
        score_key,
        dataset,
        search,
        gies,
        order,
    })
}

fn run_iflop_order_search(scorer: &mut GiesScorer<'_>, config: &SearchConfig) -> NativeResult<Candidate> {
    let p = scorer.dataset.p;
    let mut rng = SimpleRng::new(config.random_seed);
    let mut best_order = resolve_gies_initial_order(scorer.dataset, &mut rng);
    let mut best_state: Option<GiesSearchState> = None;
    let mut best_score = f64::INFINITY;
    let k = config
        .perturbation_size
        .unwrap_or_else(|| default_perturbation_size(p));
    for iteration in 0..=config.ils_restarts {
        let order = if iteration == 0 {
            best_order.clone()
        } else {
            swap_perturb_order(&best_order, &mut rng, k)
        };
        let mut state = perm_to_gies_state(scorer, &order, config.atol)?;
        local_search_gies_original(scorer, &mut state, config)?;
        state = perm_to_gies_state(scorer, &state.order, config.atol)?;
        if best_score - state.total_score > config.atol || best_state.is_none() {
            best_score = state.total_score;
            best_order = state.order.clone();
            best_state = Some(state);
        }
    }
    best_state
        .map(|state| candidate_from_gies_state(&state, scorer.dataset.p))
        .ok_or_else(|| NativeError("search produced no candidate".to_string()))
}

fn parent_sets_for_order_gies(
    scorer: &mut GiesScorer<'_>,
    order: &[usize],
    atol: f64,
) -> NativeResult<Vec<Vec<usize>>> {
    let mut parents = vec![Vec::<usize>::new(); scorer.dataset.p];
    for pos in 0..order.len() {
        let node = order[pos];
        parents[node] = fit_parents_gies(scorer, node, &order[..pos], atol)?.parents;
    }
    Ok(parents)
}

fn local_gies_result(
    scorer: &mut GiesScorer<'_>,
    node: usize,
    parents: &[usize],
) -> NativeResult<GiesNodeState> {
    scorer.local_state(node, parents)
}

fn gies_local_score_plus(
    scorer: &mut GiesScorer<'_>,
    local: &GiesNodeState,
    parent: usize,
) -> NativeResult<GiesNodeState> {
    scorer.local_state_plus(local, parent)
}

fn gies_local_score_minus(
    scorer: &mut GiesScorer<'_>,
    local: &GiesNodeState,
    parent: usize,
) -> NativeResult<GiesNodeState> {
    scorer.local_state_minus(local, parent)
}

fn set_minus(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .copied()
        .filter(|item| !right.contains(item))
        .collect()
}

fn grow_gies(
    scorer: &mut GiesScorer<'_>,
    local: &mut GiesNodeState,
    non_parents: &mut Vec<usize>,
    atol: f64,
) -> NativeResult<()> {
    loop {
        let mut any_change = false;
        for candidate in non_parents.clone().iter().rev() {
            let new_local = gies_local_score_plus(scorer, local, *candidate)?;
            if new_local.score <= local.score + atol {
                *local = new_local;
                if let Some(pos) = non_parents.iter().position(|node| node == candidate) {
                    non_parents.remove(pos);
                }
                any_change = true;
            }
        }
        if !any_change {
            break;
        }
    }
    Ok(())
}

fn shrink_gies(
    scorer: &mut GiesScorer<'_>,
    local: &mut GiesNodeState,
    non_parents: &mut Vec<usize>,
    atol: f64,
) -> NativeResult<()> {
    loop {
        let mut any_change = false;
        for parent in local.parents.clone().iter().rev() {
            let new_local = gies_local_score_minus(scorer, local, *parent)?;
            if new_local.score <= local.score + atol {
                *local = new_local;
                non_parents.push(*parent);
                any_change = true;
            }
        }
        if !any_change {
            break;
        }
    }
    Ok(())
}

fn fit_parents_gies(
    scorer: &mut GiesScorer<'_>,
    node: usize,
    prefix: &[usize],
    atol: f64,
) -> NativeResult<GiesNodeState> {
    let mut local = local_gies_result(scorer, node, &[])?;
    let mut non_parents: Vec<usize> = prefix.iter().copied().filter(|item| *item != node).collect();
    loop {
        let old_score = local.score;
        grow_gies(scorer, &mut local, &mut non_parents, atol)?;
        shrink_gies(scorer, &mut local, &mut non_parents, atol)?;
        if old_score - local.score <= atol {
            break;
        }
    }
    Ok(local)
}

fn fit_parents_gies_plus(
    scorer: &mut GiesScorer<'_>,
    mut local: GiesNodeState,
    prefix: &[usize],
    new_parents: &[usize],
    atol: f64,
) -> NativeResult<GiesNodeState> {
    for parent in new_parents {
        if !local.parents.contains(parent) {
            local = gies_local_score_plus(scorer, &local, *parent)?;
        }
    }
    let mut non_parents = set_minus(prefix, &local.parents);
    loop {
        let old_score = local.score;
        grow_gies(scorer, &mut local, &mut non_parents, atol)?;
        shrink_gies(scorer, &mut local, &mut non_parents, atol)?;
        if old_score - local.score <= atol {
            break;
        }
    }
    Ok(local)
}

fn fit_parents_gies_minus(
    scorer: &mut GiesScorer<'_>,
    mut local: GiesNodeState,
    prefix: &[usize],
    remove_parent: usize,
    atol: f64,
) -> NativeResult<GiesNodeState> {
    if !local.parents.contains(&remove_parent) {
        return Ok(local);
    }
    local = gies_local_score_minus(scorer, &local, remove_parent)?;
    let mut non_parents = set_minus(prefix, &local.parents);
    loop {
        let old_score = local.score;
        grow_gies(scorer, &mut local, &mut non_parents, atol)?;
        shrink_gies(scorer, &mut local, &mut non_parents, atol)?;
        if old_score - local.score <= atol {
            break;
        }
    }
    Ok(local)
}

fn perm_to_gies_state(
    scorer: &mut GiesScorer<'_>,
    order: &[usize],
    atol: f64,
) -> NativeResult<GiesSearchState> {
    let mut local_scores = (0..order.len())
        .map(|node| GiesNodeState {
            node,
            parents: Vec::new(),
            score: f64::INFINITY,
            env_states: Vec::new(),
        })
        .collect::<Vec<_>>();
    for idx in 0..order.len() {
        let node = order[idx];
        local_scores[node] = fit_parents_gies(scorer, node, &order[..idx], atol)?;
    }
    let total_score = local_scores.iter().map(|local| local.score).sum();
    Ok(GiesSearchState {
        order: order.to_vec(),
        local_scores,
        total_score,
    })
}

fn reinsert_gies_original(
    scorer: &mut GiesScorer<'_>,
    state: &mut GiesSearchState,
    node: usize,
    atol: f64,
) -> NativeResult<()> {
    let v_index = state
        .order
        .iter()
        .position(|item| *item == node)
        .ok_or_else(|| NativeError(format!("node {} not found in current order", node)))?;
    let mut v_curr_local = state.local_scores[node].clone();
    let p = state.order.len();
    let mut best_diff = atol;
    let mut best_ins_pos = v_index;
    let mut curr_diff = 0.0;
    let mut v_best_local: Vec<Option<GiesNodeState>> = vec![None; p];
    let mut z_best_local: Vec<Option<GiesNodeState>> = vec![None; p];

    for pos in (0..v_index).rev() {
        let z = state.order[pos];
        let mut prefix = state.order[0..pos].to_vec();
        let old_v_score = v_curr_local.score;
        let v_new_local = fit_parents_gies_minus(scorer, v_curr_local, &prefix, z, atol)?;
        let v_score_diff = v_new_local.score - old_v_score;
        v_curr_local = v_new_local.clone();
        prefix.push(node);
        let z_curr_local = state.local_scores[z].clone();
        let z_new_local = fit_parents_gies_plus(scorer, z_curr_local.clone(), &prefix, &[node], atol)?;
        let z_score_diff = z_new_local.score - z_curr_local.score;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    curr_diff = 0.0;
    v_curr_local = state.local_scores[node].clone();
    for pos in (v_index + 1)..p {
        let z = state.order[pos];
        let mut prefix = state.order[0..(pos + 1)].to_vec();
        remove_first_usize(&mut prefix, node);
        let old_v_score = v_curr_local.score;
        let v_new_local = fit_parents_gies_plus(scorer, v_curr_local, &prefix, &[z], atol)?;
        let v_score_diff = v_new_local.score - old_v_score;
        v_curr_local = v_new_local.clone();
        remove_first_usize(&mut prefix, z);
        let z_curr_local = state.local_scores[z].clone();
        let z_new_local = fit_parents_gies_minus(scorer, z_curr_local.clone(), &prefix, node, atol)?;
        let z_score_diff = z_new_local.score - z_curr_local.score;

        curr_diff += v_score_diff + z_score_diff;
        if curr_diff < best_diff {
            best_diff = curr_diff;
            best_ins_pos = pos;
            v_best_local[pos] = Some(v_new_local);
        }
        z_best_local[pos] = Some(z_new_local);
    }

    if best_ins_pos == v_index {
        return Ok(());
    }

    state.total_score += best_diff;
    state.local_scores[node] = v_best_local[best_ins_pos]
        .clone()
        .ok_or_else(|| NativeError("missing stored best local score for reinserted node".to_string()))?;
    if best_ins_pos < v_index {
        for (offset, z) in state.order[best_ins_pos..v_index].iter().copied().enumerate() {
            state.local_scores[z] = z_best_local[best_ins_pos + offset]
                .clone()
                .ok_or_else(|| NativeError(format!("missing updated local score for node {}", z)))?;
        }
    } else {
        for (offset, z) in state.order[(v_index + 1)..(best_ins_pos + 1)]
            .iter()
            .copied()
            .enumerate()
        {
            state.local_scores[z] = z_best_local[v_index + offset + 1]
                .clone()
                .ok_or_else(|| NativeError(format!("missing updated local score for node {}", z)))?;
        }
    }
    state.order.remove(v_index);
    state.order.insert(best_ins_pos, node);
    Ok(())
}

fn local_search_gies_original(
    scorer: &mut GiesScorer<'_>,
    state: &mut GiesSearchState,
    config: &SearchConfig,
) -> NativeResult<()> {
    let mut sweep_count = 0usize;
    loop {
        let old_score = state.total_score;
        for node in state.order.clone() {
            reinsert_gies_original(scorer, state, node, config.atol)?;
        }
        sweep_count += 1;
        if old_score - state.total_score <= config.atol {
            break;
        }
        if let Some(max_sweeps) = config.max_sweeps {
            if sweep_count >= max_sweeps {
                break;
            }
        }
    }
    Ok(())
}

fn candidate_from_gies_state(state: &GiesSearchState, p: usize) -> Candidate {
    let mut parents = vec![Vec::<usize>::new(); p];
    for local in &state.local_scores {
        parents[local.node] = local.parents.clone();
    }
    let dag_adjacency = adjacency_from_parents(&parents, p);
    let adjacency = cpdag_adjacency_from_parents(&parents, p);
    Candidate {
        order: state.order.clone(),
        score: state.total_score,
        parents,
        dag_adjacency,
        adjacency,
        adjacency_type: "cpdag".to_string(),
        score_vector: None,
    }
}

#[derive(Clone, Debug)]
struct Cholesky {
    data: Vec<f64>,
    dim: usize,
}

impl Cholesky {
    fn new(data: Vec<f64>, dim: usize) -> Self {
        Self { data, dim }
    }

    fn for_matrix_flat(mat: &[f64], dim: usize) -> Option<Self> {
        if mat.len() != dim * dim {
            return None;
        }
        if dim == 0 {
            return Some(Self::new(Vec::new(), 0));
        }
        let mut lower = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..=i {
                let mut sum = mat[i * dim + j];
                for k in 0..j {
                    sum -= lower[i * dim + k] * lower[j * dim + k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= 0.0 {
                        return None;
                    }
                    lower[i * dim + j] = sum.sqrt();
                } else {
                    let diag = lower[j * dim + j];
                    if diag <= 0.0 {
                        return None;
                    }
                    lower[i * dim + j] = sum / diag;
                }
            }
        }
        let mut packed = Vec::with_capacity(dim * (dim + 1) / 2);
        for col in 0..dim {
            for row in col..dim {
                packed.push(lower[row * dim + col]);
            }
        }
        Some(Self::new(packed, dim))
    }

    fn get_bottom_right(&self) -> f64 {
        *self.data.last().unwrap_or(&0.0)
    }

    fn forward_solve(&self, x: &mut [f64]) -> Result<(), ()> {
        let mut diag_idx = 0;
        for i in 0..self.dim {
            let diag = self.data[diag_idx];
            if diag <= 0.0 {
                return Err(());
            }
            let xi = x[i] / diag;
            x[i] = xi;
            for j in 0..(self.dim - i - 1) {
                x[i + 1 + j] -= self.data[diag_idx + 1 + j] * xi;
            }
            diag_idx += self.dim - i;
        }
        Ok(())
    }

    fn make_givens(a: f64, b: f64) -> (f64, f64, f64) {
        let mut c = 1.0;
        let mut s = 0.0;
        if b != 0.0 {
            if b.abs() > a.abs() {
                let tau = -a / b;
                s = -1.0 / (1.0 + tau * tau).sqrt();
                c = s * tau;
            } else {
                let tau = -b / a;
                c = 1.0 / (1.0 + tau * tau).sqrt();
                s = c * tau;
            }
        }
        let mut r = c * a - s * b;
        if r < 0.0 {
            c = -c;
            s = -s;
            r = -r;
        }
        (c, s, r)
    }

    fn insert_column_before_last(&self, mut x: Vec<f64>) -> Option<Self> {
        let new_size = (self.dim + 1) * (self.dim + 2) / 2;
        let mut new_data = vec![0.0; new_size];
        self.forward_solve(&mut x).ok()?;
        let sum: f64 = x[0..self.dim].iter().map(|value| value * value).sum();
        let new_diag_squared = x[self.dim] - sum;
        if !new_diag_squared.is_finite() || new_diag_squared <= 0.0 {
            return None;
        }
        x[self.dim] = new_diag_squared.sqrt();

        let n = x.len();
        let (c, s, r) = Self::make_givens(x[n - 2], x[n - 1]);
        x[n - 2] = r;
        x[n - 1] = 0.0;
        let prev_corner = *self.data.last()?;
        let new_left_of_corner = c * prev_corner;
        let new_corner = (s * prev_corner).abs();

        let mut src_idx = 0;
        for i in 0..self.dim {
            let stride = self.dim - i - 1;
            let dst_idx = src_idx + i;
            for offset in 0..stride {
                new_data[dst_idx + offset] = self.data[src_idx + offset];
            }
            new_data[dst_idx + stride] = x[i];
            new_data[dst_idx + stride + 1] = self.data[src_idx + stride];
            src_idx += stride + 1;
        }
        new_data[new_size - 2] = new_left_of_corner;
        new_data[new_size - 1] = new_corner;
        Some(Self::new(new_data, self.dim + 1))
    }

    fn remove_column(&self, k: usize) -> Self {
        if self.dim <= 1 {
            return Self::new(Vec::new(), 0);
        }
        let new_size = (self.dim - 1) * self.dim / 2;
        let mut new_data = vec![0.0; new_size];
        let mut x = Vec::with_capacity(self.dim.saturating_sub(k));
        let mut idx = 0usize;
        for i in 0..self.dim {
            if i < k {
                let stride = k - i;
                let dst = idx - i;
                for offset in 0..stride {
                    new_data[dst + offset] = self.data[idx + offset];
                }
                for offset in 0..(self.dim - k - 1) {
                    new_data[dst + stride + offset] = self.data[idx + stride + 1 + offset];
                }
                idx += self.dim - i;
            } else if i == k {
                idx += 1;
                let stride = self.dim - i - 1;
                x.extend_from_slice(&self.data[idx..idx + stride]);
                idx += stride;
            } else {
                let (c, s, r) = Self::make_givens(self.data[idx], x[i - k - 1]);
                new_data[idx - self.dim] = r;
                x[i - k - 1] = 0.0;
                idx += 1;
                for j in (i + 1)..self.dim {
                    let tau1 = self.data[idx];
                    let tau2 = x[j - k - 1];
                    new_data[idx - self.dim] = c * tau1 - s * tau2;
                    x[j - k - 1] = s * tau1 + c * tau2;
                    idx += 1;
                }
            }
        }
        Self::new(new_data, self.dim - 1)
    }
}

fn precompute_env_scatters(dataset: &Dataset) -> Vec<Vec<f64>> {
    dataset.envs.iter().map(centered_scatter).collect()
}

fn precompute_observational_scatter(dataset: &Dataset) -> Option<(usize, Vec<f64>)> {
    let obs_indices = dataset.observational_env_indices();
    if obs_indices.is_empty() {
        return None;
    }
    let matrix = dataset.stacked_rows(&obs_indices);
    Some((matrix.n, centered_scatter_matrix(&matrix)))
}

fn precompute_pooled_scatters_by_node(
    dataset: &Dataset,
    env_scatters: &[Vec<f64>],
    mode: LocalBicScoreMode,
) -> Vec<Option<(usize, Vec<f64>)>> {
    let mut out = Vec::<Option<(usize, Vec<f64>)>>::with_capacity(dataset.p);
    for node in 0..dataset.p {
        let env_indices = if mode == LocalBicScoreMode::FlopEnvwise {
            (0..dataset.envs.len()).collect::<Vec<_>>()
        } else {
            dataset.effective_env_indices(node)
        };
        if env_indices.is_empty() {
            out.push(None);
            continue;
        }
        let mut n_fit = 0usize;
        let mut pooled_scatter = vec![0.0; dataset.p * dataset.p];
        for idx in env_indices {
            n_fit += dataset.envs[idx].n;
            let scatter = &env_scatters[idx];
            for pos in 0..pooled_scatter.len() {
                pooled_scatter[pos] += scatter[pos];
            }
        }
        out.push(Some((n_fit, pooled_scatter)));
    }
    out
}

fn concrete_env_indices(states: &[EnvLocalState]) -> Vec<usize> {
    states
        .iter()
        .filter_map(|state| {
            if state.env_index == usize::MAX {
                None
            } else {
                Some(state.env_index)
            }
        })
        .collect()
}

fn cholesky_for_node_parents(
    scatter: &[f64],
    p: usize,
    node: usize,
    parents: &[usize],
) -> Option<Cholesky> {
    let mut items = parents.to_vec();
    items.push(node);
    let dim = items.len();
    let mut mat = vec![0.0; dim * dim];
    for (row, item_row) in items.iter().enumerate() {
        for (col, item_col) in items.iter().enumerate() {
            mat[row * dim + col] = scatter[*item_row * p + *item_col];
        }
    }
    Cholesky::for_matrix_flat(&mat, dim)
}

fn column_subvector_flat(scatter: &[f64], p: usize, items: &[usize], col: usize) -> Vec<f64> {
    items.iter().map(|item| scatter[*item * p + col]).collect()
}

fn centered_scatter(env: &EnvData) -> Vec<f64> {
    let mut means = vec![0.0; env.p];
    for row in 0..env.n {
        for col in 0..env.p {
            means[col] += env.values[row * env.p + col];
        }
    }
    for value in &mut means {
        *value /= env.n.max(1) as f64;
    }
    let mut scatter = vec![0.0; env.p * env.p];
    for row in 0..env.n {
        for a in 0..env.p {
            let za = env.values[row * env.p + a] - means[a];
            for b in 0..env.p {
                let zb = env.values[row * env.p + b] - means[b];
                scatter[a * env.p + b] += za * zb;
            }
        }
    }
    scatter
}

fn centered_scatter_matrix(matrix: &Matrix) -> Vec<f64> {
    let mut means = vec![0.0; matrix.p];
    for row in 0..matrix.n {
        for col in 0..matrix.p {
            means[col] += matrix.values[row * matrix.p + col];
        }
    }
    for value in &mut means {
        *value /= matrix.n.max(1) as f64;
    }
    let mut scatter = vec![0.0; matrix.p * matrix.p];
    for row in 0..matrix.n {
        for a in 0..matrix.p {
            let za = matrix.values[row * matrix.p + a] - means[a];
            for b in 0..matrix.p {
                let zb = matrix.values[row * matrix.p + b] - means[b];
                scatter[a * matrix.p + b] += za * zb;
            }
        }
    }
    scatter
}

fn residual_variance_scatter(
    scatter: &[f64],
    n_samples: usize,
    p: usize,
    node: usize,
    parents: &[usize],
    eps: f64,
) -> f64 {
    let n = n_samples.max(1) as f64;
    let s_yy = scatter[node * p + node];
    if parents.is_empty() {
        return (s_yy / n).max(eps);
    }
    let k = parents.len();
    let mut s_xx = vec![vec![0.0; k]; k];
    let mut s_xy = vec![0.0; k];
    for (i, parent_i) in parents.iter().enumerate() {
        s_xy[i] = scatter[*parent_i * p + node];
        for (j, parent_j) in parents.iter().enumerate() {
            s_xx[i][j] = scatter[*parent_i * p + *parent_j];
        }
    }
    let beta = gaussian_solve(&s_xx, &s_xy).or_else(|| {
        let mut ridged = s_xx.clone();
        for (idx, row) in ridged.iter_mut().enumerate() {
            row[idx] += eps;
        }
        gaussian_solve(&ridged, &s_xy)
    });
    let rss = if let Some(beta_value) = beta {
        s_yy - dot(&s_xy, &beta_value)
    } else {
        s_yy
    };
    (rss / n).max(eps)
}

fn shared_beta_from_scatter(
    scatter: &[f64],
    p: usize,
    node: usize,
    parents: &[usize],
    eps: f64,
) -> Vec<f64> {
    if parents.is_empty() {
        return Vec::new();
    }
    let k = parents.len();
    let mut s_xx = vec![vec![0.0; k]; k];
    let mut s_xy = vec![0.0; k];
    for (i, parent_i) in parents.iter().enumerate() {
        s_xy[i] = scatter[*parent_i * p + node];
        for (j, parent_j) in parents.iter().enumerate() {
            s_xx[i][j] = scatter[*parent_i * p + *parent_j];
        }
    }
    gaussian_solve(&s_xx, &s_xy)
        .or_else(|| {
            let mut ridged = s_xx.clone();
            for (idx, row) in ridged.iter_mut().enumerate() {
                row[idx] += eps;
            }
            gaussian_solve(&ridged, &s_xy)
        })
        .unwrap_or_else(|| vec![0.0; k])
}

fn residual_variance_with_beta(
    scatter: &[f64],
    n_samples: usize,
    p: usize,
    node: usize,
    parents: &[usize],
    beta: &[f64],
    eps: f64,
) -> f64 {
    let n = n_samples.max(1) as f64;
    let s_yy = scatter[node * p + node];
    if parents.is_empty() {
        return (s_yy / n).max(eps);
    }
    let mut beta_sxy = 0.0;
    let mut beta_sxx_beta = 0.0;
    for (i, parent_i) in parents.iter().enumerate() {
        beta_sxy += beta[i] * scatter[*parent_i * p + node];
        for (j, parent_j) in parents.iter().enumerate() {
            beta_sxx_beta += beta[i] * scatter[*parent_i * p + *parent_j] * beta[j];
        }
    }
    let rss = s_yy - 2.0 * beta_sxy + beta_sxx_beta;
    (rss / n).max(eps)
}

fn gaussian_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > pivot_abs {
                pivot = row;
                pivot_abs = aug[row][col].abs();
            }
        }
        if pivot_abs < 1.0e-12 {
            return None;
        }
        if pivot != col {
            aug.swap(pivot, col);
        }
        let diag = aug[col][col];
        for item in col..=n {
            aug[col][item] /= diag;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0.0 {
                continue;
            }
            for item in col..=n {
                aug[row][item] -= factor * aug[col][item];
            }
        }
    }
    Some((0..n).map(|idx| aug[idx][n]).collect())
}

fn adjacency_from_parents(parents: &[Vec<usize>], p: usize) -> Vec<Vec<u8>> {
    let mut adjacency = vec![vec![0u8; p]; p];
    for child in 0..p {
        for parent in &parents[child] {
            if *parent != child {
                adjacency[*parent][child] = 1;
            }
        }
    }
    adjacency
}

fn cpdag_adjacency_from_parents(parents: &[Vec<usize>], p: usize) -> Vec<Vec<u8>> {
    let dag = adjacency_from_parents(parents, p);
    let top_order = topological_ordering_from_parents(parents, p);
    let mut ordering = vec![0usize; p];
    for (idx, node) in top_order.iter().copied().enumerate() {
        ordering[node] = idx;
    }

    let mut edges = Vec::<(usize, usize)>::new();
    for parent in 0..p {
        for child in 0..p {
            if dag[parent][child] != 0 {
                edges.push((parent, child));
            }
        }
    }
    edges.sort_by(|(x, y), (x2, y2)| {
        let y_order = ordering[*y].cmp(&ordering[*y2]);
        if y_order == std::cmp::Ordering::Equal {
            ordering[*x2].cmp(&ordering[*x])
        } else {
            y_order
        }
    });

    let mut edge_types = vec![vec![0u8; p]; p];
    for (x, y) in edges {
        if edge_types[x][y] != 0 {
            continue;
        }
        let mut parents_y: Vec<usize> = (0..p).filter(|node| dag[*node][y] != 0).collect();
        parents_y.sort_unstable();
        let mut all_adjacent = true;
        for w in 0..p {
            if edge_types[w][x] == 1 {
                if dag[w][y] == 0 {
                    edge_types[x][y] = 1;
                    all_adjacent = false;
                    break;
                } else {
                    edge_types[w][y] = 1;
                }
            }
        }
        if !all_adjacent {
            continue;
        }
        let mut parents_x: Vec<usize> = (0..p).filter(|node| dag[*node][x] != 0).collect();
        parents_x.push(x);
        parents_x.sort_unstable();
        if parents_y != parents_x {
            for z in parents_y {
                if z != x {
                    edge_types[z][y] = 1;
                }
            }
            edge_types[x][y] = 1;
        } else {
            edge_types[x][y] = 2;
        }
    }

    let mut cpdag = vec![vec![0u8; p]; p];
    for x in 0..p {
        for y in 0..p {
            match edge_types[x][y] {
                1 => cpdag[x][y] = 1,
                2 => {
                    cpdag[x][y] = 2;
                    cpdag[y][x] = 2;
                }
                _ => {}
            }
        }
    }
    cpdag
}

fn icpdag_adjacency(dag: &[Vec<u8>], dataset: &Dataset) -> Vec<Vec<u8>> {
    let p = dataset.p;
    let mut parents = vec![Vec::<usize>::new(); p];
    for parent in 0..p {
        for child in 0..p {
            if dag[parent][child] != 0 {
                parents[child].push(parent);
            }
        }
    }
    let mut graph = cpdag_adjacency_from_parents(&parents, p);
    for env in &dataset.envs {
        if env.targets.is_empty() {
            continue;
        }
        for u in 0..p {
            for v in (u + 1)..p {
                let u_targeted = env.targets.contains(&u);
                let v_targeted = env.targets.contains(&v);
                if has_undirected(&graph, u, v) && u_targeted != v_targeted {
                    orient_as_dag(&mut graph, dag, u, v);
                }
            }
        }
    }
    apply_meek_closure(&mut graph);
    graph
}

fn orient_as_dag(graph: &mut [Vec<u8>], dag: &[Vec<u8>], u: usize, v: usize) -> bool {
    if dag[u][v] != 0 {
        graph[u][v] = 1;
        graph[v][u] = 0;
        true
    } else if dag[v][u] != 0 {
        graph[v][u] = 1;
        graph[u][v] = 0;
        true
    } else {
        false
    }
}

fn has_any_edge(graph: &[Vec<u8>], u: usize, v: usize) -> bool {
    graph[u][v] != 0 || graph[v][u] != 0
}

fn has_directed(graph: &[Vec<u8>], u: usize, v: usize) -> bool {
    graph[u][v] == 1 && graph[v][u] == 0
}

fn has_undirected(graph: &[Vec<u8>], u: usize, v: usize) -> bool {
    graph[u][v] == 2 && graph[v][u] == 2
}

fn orient_partial(graph: &mut [Vec<u8>], u: usize, v: usize) -> bool {
    if !has_undirected(graph, u, v) {
        return false;
    }
    graph[u][v] = 1;
    graph[v][u] = 0;
    true
}

fn apply_meek_closure(graph: &mut [Vec<u8>]) {
    let p = graph.len();
    let mut changed = true;
    while changed {
        changed = false;
        for a in 0..p {
            for b in 0..p {
                if !has_directed(graph, a, b) {
                    continue;
                }
                for c in 0..p {
                    if c == a || c == b {
                        continue;
                    }
                    if has_undirected(graph, b, c) && !has_any_edge(graph, a, c) {
                        changed = orient_partial(graph, b, c) || changed;
                    }
                }
            }
        }

        for a in 0..p {
            for b in 0..p {
                if !has_undirected(graph, a, b) {
                    continue;
                }
                for c in 0..p {
                    if c == a || c == b {
                        continue;
                    }
                    if has_directed(graph, a, c) && has_directed(graph, c, b) {
                        changed = orient_partial(graph, a, b) || changed;
                        break;
                    }
                }
            }
        }

        for a in 0..p {
            for b in 0..p {
                if !has_undirected(graph, a, b) {
                    continue;
                }
                let mut candidates = Vec::<usize>::new();
                for c in 0..p {
                    if c != a && c != b && has_undirected(graph, a, c) && has_directed(graph, c, b) {
                        candidates.push(c);
                    }
                }
                'outer: for idx in 0..candidates.len() {
                    for jdx in (idx + 1)..candidates.len() {
                        if !has_any_edge(graph, candidates[idx], candidates[jdx]) {
                            changed = orient_partial(graph, a, b) || changed;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
}

fn topological_ordering_from_parents(parents: &[Vec<usize>], p: usize) -> Vec<usize> {
    fn visit(node: usize, parents: &[Vec<usize>], visited: &mut [bool], out: &mut Vec<usize>) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        for parent in &parents[node] {
            visit(*parent, parents, visited, out);
        }
        out.push(node);
    }

    let mut visited = vec![false; p];
    let mut out = Vec::with_capacity(p);
    for node in 0..p {
        visit(node, parents, &mut visited, &mut out);
    }
    out
}

fn remove_first_usize(values: &mut Vec<usize>, needle: usize) {
    if let Some(index) = values.iter().position(|value| *value == needle) {
        values.remove(index);
    }
}

fn swap_perturb_order(order: &[usize], rng: &mut SimpleRng, k: usize) -> Vec<usize> {
    let mut values = order.to_vec();
    if values.len() <= 1 {
        return values;
    }
    for _ in 0..k {
        let first = rng.gen_range(values.len());
        let second = rng.gen_range(values.len());
        values.swap(first, second);
    }
    values
}

fn default_perturbation_size(p: usize) -> usize {
    (p as f64).ln().round().max(0.0) as usize
}

fn resolve_gies_initial_order(dataset: &Dataset, rng: &mut SimpleRng) -> Vec<usize> {
    let obs_indices = dataset.observational_env_indices();
    if obs_indices.len() == 1 {
        let matrix = dataset.stacked_rows(&obs_indices);
        let corr = correlation_matrix(&matrix);
        if let Some(order) = pivoted_cholesky_order(&corr, dataset.p) {
            return order;
        }
    }
    rng.permutation(dataset.p)
}

fn correlation_matrix(matrix: &Matrix) -> Vec<f64> {
    let p = matrix.p;
    let n = matrix.n.max(1) as f64;
    let mut means = vec![0.0; p];
    for row in 0..matrix.n {
        for col in 0..p {
            means[col] += matrix.at(row, col);
        }
    }
    for mean_value in &mut means {
        *mean_value /= n;
    }
    let mut cov = vec![0.0; p * p];
    for row in 0..matrix.n {
        for a in 0..p {
            let va = matrix.at(row, a) - means[a];
            for b in 0..p {
                let vb = matrix.at(row, b) - means[b];
                cov[a * p + b] += va * vb / n;
            }
        }
    }
    let mut corr = vec![0.0; p * p];
    let mut std = vec![0.0; p];
    for idx in 0..p {
        std[idx] = cov[idx * p + idx].max(0.0).sqrt();
    }
    for a in 0..p {
        for b in 0..p {
            corr[a * p + b] = if a == b {
                1.0
            } else if std[a] > 0.0 && std[b] > 0.0 {
                (cov[a * p + b] / (std[a] * std[b])).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            };
        }
    }
    corr
}

fn pivoted_cholesky_order(corr: &[f64], p: usize) -> Option<Vec<usize>> {
    if p == 0 {
        return Some(Vec::new());
    }
    if p == 1 {
        return Some(vec![0]);
    }
    if corr.len() != p * p || corr.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut matrix = corr.to_vec();
    let mut order: Vec<usize> = (0..p).collect();
    let mut max_abs = f64::NEG_INFINITY;
    let mut first_pair = (0usize, 1usize);
    for i in 0..p {
        for j in (i + 1)..p {
            let value = matrix[i * p + j].abs();
            if value > max_abs {
                max_abs = value;
                first_pair = (i, j);
            }
        }
    }
    swap_rows_and_cols(&mut matrix, p, 0, first_pair.0);
    order.swap(0, first_pair.0);
    let second_index = if first_pair.1 == 0 { first_pair.0 } else { first_pair.1 };
    swap_rows_and_cols(&mut matrix, p, 1, second_index);
    order.swap(1, second_index);

    for i in 0..(p - 1) {
        let diag = matrix[i * p + i];
        if !diag.is_finite() || diag <= 0.0 {
            return None;
        }
        for j in (i + 1)..p {
            let value = matrix[i * p + j] / diag.sqrt();
            matrix[i * p + j] = value;
            let updated = matrix[j * p + j] - value * value;
            if !updated.is_finite() {
                return None;
            }
            matrix[j * p + j] = updated;
        }
        let mut best_idx = i + 1;
        let mut best_diag = matrix[best_idx * p + best_idx];
        for j in (i + 2)..p {
            let candidate = matrix[j * p + j];
            if candidate < best_diag {
                best_diag = candidate;
                best_idx = j;
            }
        }
        swap_rows_and_cols(&mut matrix, p, i + 1, best_idx);
        order.swap(i + 1, best_idx);
    }
    let final_diag = matrix[(p - 1) * p + (p - 1)];
    if !final_diag.is_finite() || final_diag <= 0.0 {
        return None;
    }
    Some(order)
}

fn swap_rows_and_cols(matrix: &mut [f64], p: usize, a: usize, b: usize) {
    if a == b {
        return;
    }
    for col in 0..p {
        matrix.swap(a * p + col, b * p + col);
    }
    for row in 0..p {
        matrix.swap(row * p + a, row * p + b);
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    fn permutation(&mut self, p: usize) -> Vec<usize> {
        let mut values: Vec<usize> = (0..p).collect();
        for idx in (1..p).rev() {
            let swap_idx = self.gen_range(idx + 1);
            values.swap(idx, swap_idx);
        }
        values
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn sorted_unique(values: &[usize]) -> Vec<usize> {
    let mut out = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn validate_order(order: &[usize], p: usize) -> NativeResult<()> {
    if order.len() != p {
        return Err(NativeError(format!(
            "order has length {}, expected {}",
            order.len(),
            p
        )));
    }
    let sorted = sorted_unique(order);
    let expected: Vec<usize> = (0..p).collect();
    if sorted != expected {
        return Err(NativeError("order is not a permutation of variables".to_string()));
    }
    Ok(())
}

fn expect_value<'a>(parts: &'a [&str], field: &str) -> NativeResult<&'a str> {
    if parts.len() != 2 {
        return Err(NativeError(format!("{} requires exactly one value", field)));
    }
    Ok(parts[1])
}

fn parse_usize(value: &str, field: &str) -> NativeResult<usize> {
    value
        .parse::<usize>()
        .map_err(|err| NativeError(format!("failed to parse {} as usize: {}", field, err)))
}

fn parse_isize(value: &str, field: &str) -> NativeResult<isize> {
    value
        .parse::<isize>()
        .map_err(|err| NativeError(format!("failed to parse {} as isize: {}", field, err)))
}

fn parse_u64(value: &str, field: &str) -> NativeResult<u64> {
    value
        .parse::<u64>()
        .map_err(|err| NativeError(format!("failed to parse {} as u64: {}", field, err)))
}

fn parse_i32(value: &str, field: &str) -> NativeResult<i32> {
    value
        .parse::<i32>()
        .map_err(|err| NativeError(format!("failed to parse {} as i32: {}", field, err)))
}

fn parse_f64(value: &str, field: &str) -> NativeResult<f64> {
    value
        .parse::<f64>()
        .map_err(|err| NativeError(format!("failed to parse {} as f64: {}", field, err)))
}

fn print_candidate(score_key: &str, candidate: &Candidate) {
    println!("IFLOP_NATIVE_RESULT_V1");
    println!("status ok");
    println!("score_key {}", score_key);
    println!("p {}", candidate.order.len());
    println!("total_score {:.17e}", candidate.score);
    println!("order {} {}", candidate.order.len(), join_usize(&candidate.order));
    println!("adjacency_type {}", candidate.adjacency_type);
    match candidate.score_vector {
        Some((edges, contradictions)) => println!("score_vector {} {}", edges, contradictions),
        None => println!("score_vector none"),
    }
    println!("parents_start");
    for (child, parents) in candidate.parents.iter().enumerate() {
        println!("parents {} {} {}", child, parents.len(), join_usize(parents));
    }
    println!("parents_end");
    println!("adjacency_start");
    for row in &candidate.adjacency {
        let text = row.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(" ");
        println!("{}", text);
    }
    println!("adjacency_end");
    println!("dag_adjacency_start");
    for row in &candidate.dag_adjacency {
        let text = row.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(" ");
        println!("{}", text);
    }
    println!("dag_adjacency_end");
    println!("END");
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn sanitize_line(value: &str) -> String {
    value.replace('\n', " ").replace('\r', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_solve_basic() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let b = vec![2.0, 8.0];
        let x = gaussian_solve(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1.0e-12);
        assert!((x[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_pivoted_cholesky_failure_falls_back_to_random_permutation() {
        let singular_corr = vec![1.0; 9];
        assert!(pivoted_cholesky_order(&singular_corr, 3).is_none());

        let dataset = Dataset {
            p: 3,
            envs: vec![EnvData {
                id: 0,
                n: 4,
                p: 3,
                values: vec![1.0; 12],
                targets: Vec::new(),
            }],
        };
        let mut expected_rng = SimpleRng::new(17);
        let expected = expected_rng.permutation(3);
        let mut actual_rng = SimpleRng::new(17);
        let actual = resolve_gies_initial_order(&dataset, &mut actual_rng);
        assert_eq!(actual, expected);
    }
}
