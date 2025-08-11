// Aether Probabilistic Programming Module
// Native support for probabilistic constructs, random variables, and Bayesian inference

use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::compiler::ast::{ASTNode, ASTNodeRef, AtomValue};
use crate::compiler::types::Distribution;

/// Random variable declaration with distribution
#[derive(Debug, Clone, PartialEq)]
pub struct RandomVariable {
    pub name: String,
    pub distribution: Distribution,
    pub observed_value: Option<ObservedValue>,
    pub dependencies: Vec<String>, // Other random variables this depends on
    pub constraints: Vec<ProbabilisticConstraint>,
}

/// Observed value for conditioning
#[derive(Debug, Clone, PartialEq)]
pub enum ObservedValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Tensor(Vec<usize>, Vec<f64>), // shape, flattened data
}

/// Probabilistic constraint for model validation
#[derive(Debug, Clone, PartialEq)]
pub enum ProbabilisticConstraint {
    /// Parameter must be positive
    Positive(String),
    /// Parameter must be in range
    Range(String, f64, f64),
    /// Sum of parameters must equal value (for categorical distributions)
    SumEquals(Vec<String>, f64),
    /// Parameters must be ordered
    Ordered(Vec<String>),
}

/// Probabilistic model definition
#[derive(Debug, Clone)]
pub struct ProbabilisticModel {
    pub name: String,
    pub random_variables: HashMap<String, RandomVariable>,
    pub observations: HashMap<String, ObservedValue>,
    pub dependencies: DependencyGraph,
    pub inference_config: InferenceConfig,
}

/// Dependency graph for probabilistic model
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    pub nodes: HashSet<String>,
    pub edges: Vec<(String, String)>, // (parent, child) relationships
    pub topological_order: Vec<String>,
}

/// Inference algorithm configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub algorithm: InferenceAlgorithm,
    pub num_samples: usize,
    pub burn_in: usize,
    pub thinning: usize,
    pub chains: usize,
    pub convergence_threshold: f64,
}

/// Available inference algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceAlgorithm {
    /// Markov Chain Monte Carlo
    MCMC {
        sampler: MCMCSampler,
        step_size: f64,
        adaptation: bool,
    },
    /// Variational Inference
    VariationalInference {
        optimizer: VIOptimizer,
        learning_rate: f64,
        max_iterations: usize,
    },
    /// Hamiltonian Monte Carlo
    HMC {
        step_size: f64,
        num_leapfrog_steps: usize,
        mass_matrix: MassMatrix,
    },
    /// No-U-Turn Sampler
    NUTS {
        step_size: f64,
        max_tree_depth: usize,
        target_accept_prob: f64,
    },
    /// Automatic algorithm selection
    Auto,
}

/// MCMC sampler types
#[derive(Debug, Clone, PartialEq)]
pub enum MCMCSampler {
    MetropolisHastings,
    Gibbs,
    SliceSampler,
    EllipticalSlice,
}

/// Variational inference optimizers
#[derive(Debug, Clone, PartialEq)]
pub enum VIOptimizer {
    ADVI, // Automatic Differentiation Variational Inference
    BBVI, // Black Box Variational Inference
    SVGD, // Stein Variational Gradient Descent
}

/// Mass matrix for HMC
#[derive(Debug, Clone, PartialEq)]
pub enum MassMatrix {
    Identity,
    Diagonal(Vec<f64>),
    Dense(Vec<Vec<f64>>),
    Adaptive,
}

/// Uncertainty propagation through computation graphs
#[derive(Debug, Clone)]
pub struct UncertaintyPropagation {
    pub variables: HashMap<String, UncertaintyInfo>,
    pub operations: Vec<UncertaintyOperation>,
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
}

/// Uncertainty information for a variable
#[derive(Debug, Clone)]
pub struct UncertaintyInfo {
    pub mean: f64,
    pub variance: f64,
    pub distribution_type: DistributionType,
    pub samples: Option<Vec<f64>>,
}

/// Type of uncertainty distribution
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionType {
    Gaussian,
    LogNormal,
    Beta,
    Gamma,
    Empirical,
}

/// Uncertainty operation for propagation
#[derive(Debug, Clone)]
pub struct UncertaintyOperation {
    pub operation_type: OperationType,
    pub inputs: Vec<String>,
    pub output: String,
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
}

/// Types of operations for uncertainty propagation
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sin,
    Cos,
    Custom(String),
}

/// Confidence interval representation
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub method: IntervalMethod,
}

/// Method for computing confidence intervals
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalMethod {
    Percentile,
    HighestDensity,
    Bootstrap,
    Analytical,
}

/// Probabilistic programming engine
pub struct ProbabilisticEngine {
    models: HashMap<String, ProbabilisticModel>,
    inference_cache: HashMap<String, InferenceResult>,
    rng_state: u64,
}

/// Result of probabilistic inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub samples: HashMap<String, Vec<f64>>,
    pub posterior_means: HashMap<String, f64>,
    pub posterior_variances: HashMap<String, f64>,
    pub log_likelihood: f64,
    pub convergence_diagnostics: ConvergenceDiagnostics,
    pub computation_time: f64,
}

/// Convergence diagnostics for MCMC
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    pub r_hat: HashMap<String, f64>, // Gelman-Rubin statistic
    pub effective_sample_size: HashMap<String, f64>,
    pub autocorrelation: HashMap<String, Vec<f64>>,
    pub trace_plots: HashMap<String, Vec<f64>>,
}

impl RandomVariable {
    /// Create new random variable
    pub fn new(name: String, distribution: Distribution) -> Self {
        RandomVariable {
            name,
            distribution,
            observed_value: None,
            dependencies: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Create observed random variable
    pub fn observed(name: String, distribution: Distribution, value: ObservedValue) -> Self {
        RandomVariable {
            name,
            distribution,
            observed_value: Some(value),
            dependencies: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add dependency on another random variable
    pub fn add_dependency(&mut self, dependency: String) {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
        }
    }

    /// Add constraint
    pub fn add_constraint(&mut self, constraint: ProbabilisticConstraint) {
        self.constraints.push(constraint);
    }

    /// Check if variable is observed
    pub fn is_observed(&self) -> bool {
        self.observed_value.is_some()
    }

    /// Get log probability density
    pub fn log_pdf(&self, value: f64) -> f64 {
        match &self.distribution {
            Distribution::Normal { mean, std } => {
                let diff = value - mean;
                -0.5 * (diff * diff) / (std * std) - 0.5 * (2.0 * std::f64::consts::PI * std * std).ln()
            }
            Distribution::Uniform { min, max } => {
                if value >= *min && value <= *max {
                    -(max - min).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Bernoulli { p } => {
                if value == 1.0 {
                    p.ln()
                } else if value == 0.0 {
                    (1.0 - p).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Categorical { probs } => {
                let index = value as usize;
                if index < probs.len() {
                    probs[index].ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Custom(_) => {
                // Custom distributions would need external implementation
                0.0
            }
        }
    }

    /// Sample from distribution
    pub fn sample(&self, rng_state: &mut u64) -> f64 {
        match &self.distribution {
            Distribution::Normal { mean, std } => {
                // Box-Muller transform for normal sampling
                let u1 = self.uniform_sample(rng_state);
                let u2 = self.uniform_sample(rng_state);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z0
            }
            Distribution::Uniform { min, max } => {
                let u = self.uniform_sample(rng_state);
                min + (max - min) * u
            }
            Distribution::Bernoulli { p } => {
                let u = self.uniform_sample(rng_state);
                if u < *p { 1.0 } else { 0.0 }
            }
            Distribution::Categorical { probs } => {
                let u = self.uniform_sample(rng_state);
                let mut cumsum = 0.0;
                for (i, &prob) in probs.iter().enumerate() {
                    cumsum += prob;
                    if u <= cumsum {
                        return i as f64;
                    }
                }
                (probs.len() - 1) as f64
            }
            Distribution::Custom(_) => {
                // Custom distributions would need external implementation
                0.0
            }
        }
    }

    /// Generate uniform random sample using linear congruential generator
    fn uniform_sample(&self, rng_state: &mut u64) -> f64 {
        // Simple LCG for demonstration
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (*rng_state as f64) / (u64::MAX as f64)
    }
}

impl ProbabilisticModel {
    /// Create new probabilistic model
    pub fn new(name: String) -> Self {
        ProbabilisticModel {
            name,
            random_variables: HashMap::new(),
            observations: HashMap::new(),
            dependencies: DependencyGraph::new(),
            inference_config: InferenceConfig::default(),
        }
    }

    /// Add random variable to model
    pub fn add_random_variable(&mut self, var: RandomVariable) {
        let name = var.name.clone();
        
        // Add dependencies to graph
        for dep in &var.dependencies {
            self.dependencies.add_edge(dep.clone(), name.clone());
        }
        
        self.dependencies.add_node(name.clone());
        self.random_variables.insert(name, var);
        
        // Update topological order
        self.dependencies.update_topological_order();
    }

    /// Add observation
    pub fn observe(&mut self, var_name: String, value: ObservedValue) {
        self.observations.insert(var_name.clone(), value.clone());
        
        // Update the random variable if it exists
        if let Some(var) = self.random_variables.get_mut(&var_name) {
            var.observed_value = Some(value);
        }
    }

    /// Get log likelihood of model given current parameter values
    pub fn log_likelihood(&self, _parameters: &HashMap<String, f64>) -> f64 {
        let mut log_likelihood = 0.0;
        
        for (_var_name, var) in &self.random_variables {
            if let Some(observed_value) = &var.observed_value {
                if let ObservedValue::Scalar(value) = observed_value {
                    // For now, assume scalar observations
                    log_likelihood += var.log_pdf(*value);
                }
            }
        }
        
        log_likelihood
    }

    /// Validate model structure
    pub fn validate(&self) -> Result<(), String> {
        // Check for cycles in dependency graph
        if self.dependencies.has_cycles() {
            return Err("Model contains cyclic dependencies".to_string());
        }
        
        // Check that all dependencies exist
        for (var_name, var) in &self.random_variables {
            for dep in &var.dependencies {
                if !self.random_variables.contains_key(dep) {
                    return Err(format!("Variable '{}' depends on undefined variable '{}'", var_name, dep));
                }
            }
        }
        
        // Validate constraints
        for var in self.random_variables.values() {
            for constraint in &var.constraints {
                if let Err(msg) = self.validate_constraint(constraint) {
                    return Err(format!("Constraint validation failed for '{}': {}", var.name, msg));
                }
            }
        }
        
        Ok(())
    }

    /// Validate individual constraint
    fn validate_constraint(&self, constraint: &ProbabilisticConstraint) -> Result<(), String> {
        match constraint {
            ProbabilisticConstraint::Positive(_param) => {
                // Would check if parameter is positive in actual implementation
                Ok(())
            }
            ProbabilisticConstraint::Range(_param, min, max) => {
                if min >= max {
                    return Err(format!("Invalid range: {} >= {}", min, max));
                }
                Ok(())
            }
            ProbabilisticConstraint::SumEquals(params, _target) => {
                if params.is_empty() {
                    return Err("Empty parameter list for SumEquals constraint".to_string());
                }
                Ok(())
            }
            ProbabilisticConstraint::Ordered(params) => {
                if params.len() < 2 {
                    return Err("Ordered constraint requires at least 2 parameters".to_string());
                }
                Ok(())
            }
        }
    }
}

impl DependencyGraph {
    /// Create new dependency graph
    pub fn new() -> Self {
        DependencyGraph {
            nodes: HashSet::new(),
            edges: Vec::new(),
            topological_order: Vec::new(),
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, node: String) {
        self.nodes.insert(node);
    }

    /// Add edge to graph
    pub fn add_edge(&mut self, from: String, to: String) {
        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());
        self.edges.push((from, to));
    }

    /// Check if graph has cycles using DFS
    pub fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for node in &self.nodes {
            if !visited.contains(node) {
                if self.has_cycle_util(node, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    /// Utility function for cycle detection
    fn has_cycle_util(&self, node: &str, visited: &mut HashSet<String>, rec_stack: &mut HashSet<String>) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        
        // Check all adjacent nodes
        for (from, to) in &self.edges {
            if from == node {
                if !visited.contains(to) {
                    if self.has_cycle_util(to, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(to) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(node);
        false
    }

    /// Update topological order using Kahn's algorithm
    pub fn update_topological_order(&mut self) {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();
        
        // Initialize in-degree and adjacency list
        for node in &self.nodes {
            in_degree.insert(node.clone(), 0);
            adj_list.insert(node.clone(), Vec::new());
        }
        
        for (from, to) in &self.edges {
            adj_list.get_mut(from).unwrap().push(to.clone());
            *in_degree.get_mut(to).unwrap() += 1;
        }
        
        // Find nodes with no incoming edges
        let mut queue: Vec<String> = in_degree.iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node, _)| node.clone())
            .collect();
        
        let mut topo_order = Vec::new();
        
        while let Some(node) = queue.pop() {
            topo_order.push(node.clone());
            
            // Reduce in-degree of adjacent nodes
            if let Some(neighbors) = adj_list.get(&node) {
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(neighbor.clone());
                    }
                }
            }
        }
        
        self.topological_order = topo_order;
    }
}

impl InferenceConfig {
    /// Create default inference configuration
    pub fn default() -> Self {
        InferenceConfig {
            algorithm: InferenceAlgorithm::Auto,
            num_samples: 1000,
            burn_in: 500,
            thinning: 1,
            chains: 4,
            convergence_threshold: 1.1, // R-hat threshold
        }
    }

    /// Create MCMC configuration
    pub fn mcmc(sampler: MCMCSampler, num_samples: usize) -> Self {
        InferenceConfig {
            algorithm: InferenceAlgorithm::MCMC {
                sampler,
                step_size: 0.1,
                adaptation: true,
            },
            num_samples,
            burn_in: num_samples / 2,
            thinning: 1,
            chains: 4,
            convergence_threshold: 1.1,
        }
    }

    /// Create variational inference configuration
    pub fn variational(optimizer: VIOptimizer, max_iterations: usize) -> Self {
        InferenceConfig {
            algorithm: InferenceAlgorithm::VariationalInference {
                optimizer,
                learning_rate: 0.01,
                max_iterations,
            },
            num_samples: 1000, // For final sampling from variational posterior
            burn_in: 0,
            thinning: 1,
            chains: 1,
            convergence_threshold: 1e-6,
        }
    }
}

impl ProbabilisticEngine {
    /// Create new probabilistic engine
    pub fn new() -> Self {
        ProbabilisticEngine {
            models: HashMap::new(),
            inference_cache: HashMap::new(),
            rng_state: 12345, // Default seed
        }
    }

    /// Add model to engine
    pub fn add_model(&mut self, model: ProbabilisticModel) -> Result<(), String> {
        model.validate()?;
        self.models.insert(model.name.clone(), model);
        Ok(())
    }

    /// Run inference on model
    pub fn infer(&mut self, model_name: &str) -> Result<InferenceResult, String> {
        // Check cache first
        if let Some(cached_result) = self.inference_cache.get(model_name) {
            return Ok(cached_result.clone());
        }

        // Clone the model to avoid borrowing issues
        let model = self.models.get(model_name)
            .ok_or_else(|| format!("Model '{}' not found", model_name))?
            .clone();

        let result = match &model.inference_config.algorithm {
            InferenceAlgorithm::MCMC { sampler, step_size, adaptation } => {
                self.run_mcmc(&model, sampler, *step_size, *adaptation)
            }
            InferenceAlgorithm::VariationalInference { optimizer, learning_rate, max_iterations } => {
                self.run_variational_inference(&model, optimizer, *learning_rate, *max_iterations)
            }
            InferenceAlgorithm::HMC { step_size, num_leapfrog_steps, mass_matrix } => {
                self.run_hmc(&model, *step_size, *num_leapfrog_steps, mass_matrix)
            }
            InferenceAlgorithm::NUTS { step_size, max_tree_depth, target_accept_prob } => {
                self.run_nuts(&model, *step_size, *max_tree_depth, *target_accept_prob)
            }
            InferenceAlgorithm::Auto => {
                self.auto_select_algorithm(&model)
            }
        }?;

        // Cache result
        self.inference_cache.insert(model_name.to_string(), result.clone());
        Ok(result)
    }

    /// Run MCMC inference
    fn run_mcmc(&mut self, model: &ProbabilisticModel, _sampler: &MCMCSampler, step_size: f64, _adaptation: bool) -> Result<InferenceResult, String> {
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        let mut current_state: HashMap<String, f64> = HashMap::new();
        
        // Initialize state
        for var_name in model.random_variables.keys() {
            if !model.observations.contains_key(var_name) {
                current_state.insert(var_name.clone(), 0.0); // Initialize to zero
                samples.insert(var_name.clone(), Vec::new());
            }
        }
        
        let total_samples = model.inference_config.num_samples + model.inference_config.burn_in;
        let mut accepted = 0;
        
        for i in 0..total_samples {
            // Metropolis-Hastings step for each variable
            for var_name in &model.dependencies.topological_order {
                if let Some(var) = model.random_variables.get(var_name) {
                    if !var.is_observed() {
                        let current_value = current_state[var_name];
                        let proposed_value = current_value + step_size * self.normal_sample();
                        
                        // Compute acceptance probability
                        let current_log_prob = var.log_pdf(current_value);
                        let proposed_log_prob = var.log_pdf(proposed_value);
                        let log_alpha = proposed_log_prob - current_log_prob;
                        
                        if log_alpha > 0.0 || self.uniform_sample() < log_alpha.exp() {
                            current_state.insert(var_name.clone(), proposed_value);
                            if i >= model.inference_config.burn_in {
                                accepted += 1;
                            }
                        }
                        
                        // Store sample after burn-in
                        if i >= model.inference_config.burn_in && i % model.inference_config.thinning == 0 {
                            samples.get_mut(var_name).unwrap().push(current_state[var_name]);
                        }
                    }
                }
            }
        }
        
        // Compute posterior statistics
        let mut posterior_means = HashMap::new();
        let mut posterior_variances = HashMap::new();
        
        for (var_name, var_samples) in &samples {
            let mean = var_samples.iter().sum::<f64>() / var_samples.len() as f64;
            let variance = var_samples.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (var_samples.len() - 1) as f64;
            
            posterior_means.insert(var_name.clone(), mean);
            posterior_variances.insert(var_name.clone(), variance);
        }
        
        let log_likelihood = model.log_likelihood(&current_state);
        
        Ok(InferenceResult {
            samples,
            posterior_means,
            posterior_variances,
            log_likelihood,
            convergence_diagnostics: ConvergenceDiagnostics::default(),
            computation_time: 0.0, // Would measure actual time
        })
    }

    /// Run variational inference
    fn run_variational_inference(&mut self, model: &ProbabilisticModel, _optimizer: &VIOptimizer, learning_rate: f64, max_iterations: usize) -> Result<InferenceResult, String> {
        // Simplified variational inference implementation
        // In practice, this would use automatic differentiation
        
        let mut variational_params: HashMap<String, (f64, f64)> = HashMap::new(); // (mean, log_std)
        
        // Initialize variational parameters
        for var_name in model.random_variables.keys() {
            if !model.observations.contains_key(var_name) {
                variational_params.insert(var_name.clone(), (0.0, 0.0)); // mean=0, std=1
            }
        }
        
        // Optimization loop
        for _iteration in 0..max_iterations {
            // Compute ELBO gradient (simplified)
            for (_var_name, (mean, log_std)) in &mut variational_params {
                // Gradient ascent step (simplified)
                *mean += learning_rate * 0.01; // Placeholder gradient
                *log_std += learning_rate * 0.001; // Placeholder gradient
            }
        }
        
        // Sample from variational posterior
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        let mut posterior_means = HashMap::new();
        let mut posterior_variances = HashMap::new();
        
        for (var_name, (mean, log_std)) in &variational_params {
            let std = log_std.exp();
            let mut var_samples = Vec::new();
            
            for _ in 0..model.inference_config.num_samples {
                let sample = mean + std * self.normal_sample();
                var_samples.push(sample);
            }
            
            samples.insert(var_name.clone(), var_samples);
            posterior_means.insert(var_name.clone(), *mean);
            posterior_variances.insert(var_name.clone(), std * std);
        }
        
        Ok(InferenceResult {
            samples,
            posterior_means,
            posterior_variances,
            log_likelihood: 0.0, // Would compute actual ELBO
            convergence_diagnostics: ConvergenceDiagnostics::default(),
            computation_time: 0.0,
        })
    }

    /// Run Hamiltonian Monte Carlo
    fn run_hmc(&mut self, model: &ProbabilisticModel, step_size: f64, _num_leapfrog_steps: usize, _mass_matrix: &MassMatrix) -> Result<InferenceResult, String> {
        // Simplified HMC implementation
        // Would need gradient computation for full implementation
        self.run_mcmc(model, &MCMCSampler::MetropolisHastings, step_size, false)
    }

    /// Run No-U-Turn Sampler
    fn run_nuts(&mut self, model: &ProbabilisticModel, step_size: f64, _max_tree_depth: usize, _target_accept_prob: f64) -> Result<InferenceResult, String> {
        // Simplified NUTS implementation
        // Would need full tree building and trajectory sampling
        self.run_mcmc(model, &MCMCSampler::MetropolisHastings, step_size, true)
    }

    /// Automatically select inference algorithm
    fn auto_select_algorithm(&mut self, model: &ProbabilisticModel) -> Result<InferenceResult, String> {
        let num_vars = model.random_variables.len();
        let has_continuous = model.random_variables.values()
            .any(|var| matches!(var.distribution, Distribution::Normal { .. } | Distribution::Uniform { .. }));
        
        if num_vars < 10 && has_continuous {
            // Use NUTS for small continuous models
            self.run_nuts(model, 0.1, 10, 0.8)
        } else if has_continuous {
            // Use HMC for larger continuous models
            self.run_hmc(model, 0.1, 10, &MassMatrix::Identity)
        } else {
            // Use Gibbs sampling for discrete models
            self.run_mcmc(model, &MCMCSampler::Gibbs, 0.1, false)
        }
    }

    /// Generate normal random sample
    fn normal_sample(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.uniform_sample();
        let u2 = self.uniform_sample();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate uniform random sample
    fn uniform_sample(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Propagate uncertainty through computation graph
    pub fn propagate_uncertainty(&self, model_name: &str, operations: Vec<UncertaintyOperation>) -> Result<UncertaintyPropagation, String> {
        let inference_result = self.inference_cache.get(model_name)
            .ok_or_else(|| format!("No inference results found for model '{}'", model_name))?;

        let mut variables: HashMap<String, UncertaintyInfo> = HashMap::new();
        let mut confidence_intervals: HashMap<String, ConfidenceInterval> = HashMap::new();

        // Initialize uncertainty info from inference results
        for (var_name, samples) in &inference_result.samples {
            let mean = inference_result.posterior_means[var_name];
            let variance = inference_result.posterior_variances[var_name];
            
            variables.insert(var_name.clone(), UncertaintyInfo {
                mean,
                variance,
                distribution_type: DistributionType::Empirical,
                samples: Some(samples.clone()),
            });

            // Compute confidence interval
            let mut sorted_samples = samples.clone();
            sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted_samples.len();
            let lower_idx = (n as f64 * 0.025) as usize;
            let upper_idx = (n as f64 * 0.975) as usize;
            
            confidence_intervals.insert(var_name.clone(), ConfidenceInterval {
                lower: sorted_samples[lower_idx],
                upper: sorted_samples[upper_idx],
                confidence_level: 0.95,
                method: IntervalMethod::Percentile,
            });
        }

        // Propagate uncertainty through operations
        for operation in &operations {
            match operation.operation_type {
                OperationType::Add => {
                    if operation.inputs.len() == 2 {
                        let var1 = &variables[&operation.inputs[0]];
                        let var2 = &variables[&operation.inputs[1]];
                        
                        let result_mean = var1.mean + var2.mean;
                        let result_variance = var1.variance + var2.variance; // Assuming independence
                        
                        variables.insert(operation.output.clone(), UncertaintyInfo {
                            mean: result_mean,
                            variance: result_variance,
                            distribution_type: DistributionType::Gaussian,
                            samples: None,
                        });
                    }
                }
                OperationType::Multiply => {
                    if operation.inputs.len() == 2 {
                        let var1 = &variables[&operation.inputs[0]];
                        let var2 = &variables[&operation.inputs[1]];
                        
                        // Delta method approximation
                        let result_mean = var1.mean * var2.mean;
                        let result_variance = var1.variance * var2.mean.powi(2) + 
                                            var2.variance * var1.mean.powi(2) +
                                            var1.variance * var2.variance;
                        
                        variables.insert(operation.output.clone(), UncertaintyInfo {
                            mean: result_mean,
                            variance: result_variance,
                            distribution_type: DistributionType::LogNormal,
                            samples: None,
                        });
                    }
                }
                _ => {
                    // Other operations would be implemented similarly
                }
            }
        }

        Ok(UncertaintyPropagation {
            variables,
            operations: operations.clone(),
            confidence_intervals,
        })
    }
}

impl ConvergenceDiagnostics {
    /// Create default convergence diagnostics
    pub fn default() -> Self {
        ConvergenceDiagnostics {
            r_hat: HashMap::new(),
            effective_sample_size: HashMap::new(),
            autocorrelation: HashMap::new(),
            trace_plots: HashMap::new(),
        }
    }

    /// Compute R-hat statistic for convergence assessment
    pub fn compute_r_hat(&mut self, chains: &[Vec<f64>]) -> f64 {
        if chains.len() < 2 {
            return 1.0; // Cannot compute R-hat with single chain
        }

        let n = chains[0].len();
        let m = chains.len();
        
        // Between-chain variance
        let chain_means: Vec<f64> = chains.iter()
            .map(|chain| chain.iter().sum::<f64>() / n as f64)
            .collect();
        let overall_mean = chain_means.iter().sum::<f64>() / m as f64;
        let b = (n as f64) * chain_means.iter()
            .map(|mean| (mean - overall_mean).powi(2))
            .sum::<f64>() / (m - 1) as f64;
        
        // Within-chain variance
        let w = chains.iter().zip(&chain_means)
            .map(|(chain, &chain_mean)| {
                chain.iter().map(|x| (x - chain_mean).powi(2)).sum::<f64>() / (n - 1) as f64
            })
            .sum::<f64>() / m as f64;
        
        // Marginal posterior variance
        let var_plus = ((n - 1) as f64 * w + b) / n as f64;
        
        // R-hat
        (var_plus / w).sqrt()
    }

    /// Compute effective sample size
    pub fn compute_effective_sample_size(&mut self, samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let autocorr = self.compute_autocorrelation(samples);
        
        // Sum autocorrelations until they become negligible
        let mut sum_autocorr = 1.0; // lag 0 autocorrelation is 1
        for (_i, &corr) in autocorr.iter().enumerate().skip(1) {
            if corr < 0.05 { // Threshold for negligible correlation
                break;
            }
            sum_autocorr += 2.0 * corr; // Factor of 2 for positive and negative lags
        }
        
        n / sum_autocorr
    }

    /// Compute autocorrelation function
    pub fn compute_autocorrelation(&self, samples: &[f64]) -> Vec<f64> {
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        let max_lag = (n / 4).min(100); // Limit maximum lag
        let mut autocorr = Vec::with_capacity(max_lag);
        
        for lag in 0..max_lag {
            let mut covariance = 0.0;
            let count = n - lag;
            
            for i in 0..count {
                covariance += (samples[i] - mean) * (samples[i + lag] - mean);
            }
            
            covariance /= count as f64;
            autocorr.push(covariance / variance);
        }
        
        autocorr
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distribution::Normal { mean, std } => write!(f, "Normal({}, {})", mean, std),
            Distribution::Uniform { min, max } => write!(f, "Uniform({}, {})", min, max),
            Distribution::Bernoulli { p } => write!(f, "Bernoulli({})", p),
            Distribution::Categorical { probs } => write!(f, "Categorical({:?})", probs),
            Distribution::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl fmt::Display for InferenceAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceAlgorithm::MCMC { sampler, .. } => write!(f, "MCMC({:?})", sampler),
            InferenceAlgorithm::VariationalInference { optimizer, .. } => write!(f, "VI({:?})", optimizer),
            InferenceAlgorithm::HMC { .. } => write!(f, "HMC"),
            InferenceAlgorithm::NUTS { .. } => write!(f, "NUTS"),
            InferenceAlgorithm::Auto => write!(f, "Auto"),
        }
    }
}

// Syntax parsing support for probabilistic constructs
pub fn parse_random_variable_declaration(ast: &ASTNode) -> Result<RandomVariable, String> {
    // Parse syntax like: (~ weight (Normal 0.0 1.0))
    match ast {
        ASTNode::List(elements) => {
            if elements.len() >= 3 {
                // Extract nodes from references
                let nodes: Vec<&ASTNode> = elements.iter()
                    .filter_map(|node_ref| match node_ref {
                        ASTNodeRef::Direct(node) => Some(node.as_ref()),
                        _ => None,
                    })
                    .collect();
                
                if nodes.len() >= 3 {
                    if let (ASTNode::Atom(AtomValue::Symbol(op)),
                            ASTNode::Atom(AtomValue::Symbol(var_name)),
                            ASTNode::List(dist_elements)) = (nodes[0], nodes[1], nodes[2]) {
                        
                        if op == "~" && !dist_elements.is_empty() {
                            let dist_nodes: Vec<&ASTNode> = dist_elements.iter()
                                .filter_map(|node_ref| match node_ref {
                                    ASTNodeRef::Direct(node) => Some(node.as_ref()),
                                    _ => None,
                                })
                                .collect();
                            
                            if let ASTNode::Atom(AtomValue::Symbol(dist_name)) = dist_nodes[0] {
                                let distribution = match dist_name.as_str() {
                                    "Normal" => {
                                        if dist_nodes.len() >= 3 {
                                            if let (ASTNode::Atom(AtomValue::Number(mean)),
                                                    ASTNode::Atom(AtomValue::Number(std))) =
                                                (dist_nodes[1], dist_nodes[2]) {
                                                Distribution::Normal { mean: *mean, std: *std }
                                            } else {
                                                return Err("Invalid Normal distribution parameters".to_string());
                                            }
                                        } else {
                                            return Err("Normal distribution requires mean and std parameters".to_string());
                                        }
                                    }
                                    "Uniform" => {
                                        if dist_nodes.len() >= 3 {
                                            if let (ASTNode::Atom(AtomValue::Number(min)),
                                                    ASTNode::Atom(AtomValue::Number(max))) =
                                                (dist_nodes[1], dist_nodes[2]) {
                                                Distribution::Uniform { min: *min, max: *max }
                                            } else {
                                                return Err("Invalid Uniform distribution parameters".to_string());
                                            }
                                        } else {
                                            return Err("Uniform distribution requires min and max parameters".to_string());
                                        }
                                    }
                                    "Bernoulli" => {
                                        if dist_nodes.len() >= 2 {
                                            if let ASTNode::Atom(AtomValue::Number(p)) = dist_nodes[1] {
                                                Distribution::Bernoulli { p: *p }
                                            } else {
                                                return Err("Invalid Bernoulli distribution parameter".to_string());
                                            }
                                        } else {
                                            return Err("Bernoulli distribution requires probability parameter".to_string());
                                        }
                                    }
                                    _ => {
                                        return Err(format!("Unknown distribution: {}", dist_name));
                                    }
                                };
                                
                                return Ok(RandomVariable::new(var_name.clone(), distribution));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    
    Err("Invalid random variable declaration syntax".to_string())
}

pub fn parse_observation(ast: &ASTNode) -> Result<(String, ObservedValue), String> {
    // Parse syntax like: (observe weight 0.5)
    match ast {
        ASTNode::List(elements) => {
            if elements.len() >= 3 {
                let nodes: Vec<&ASTNode> = elements.iter()
                    .filter_map(|node_ref| match node_ref {
                        ASTNodeRef::Direct(node) => Some(node.as_ref()),
                        _ => None,
                    })
                    .collect();
                
                if nodes.len() >= 3 {
                    if let (ASTNode::Atom(AtomValue::Symbol(op)),
                            ASTNode::Atom(AtomValue::Symbol(var_name)),
                            ASTNode::Atom(AtomValue::Number(value))) = (nodes[0], nodes[1], nodes[2]) {
                        
                        if op == "observe" {
                            return Ok((var_name.clone(), ObservedValue::Scalar(*value)));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    
    Err("Invalid observation syntax".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_variable_creation() {
        let var = RandomVariable::new(
            "weight".to_string(),
            Distribution::Normal { mean: 0.0, std: 1.0 }
        );
        
        assert_eq!(var.name, "weight");
        assert!(!var.is_observed());
        assert_eq!(var.dependencies.len(), 0);
    }

    #[test]
    fn test_observed_variable() {
        let var = RandomVariable::observed(
            "data".to_string(),
            Distribution::Normal { mean: 0.0, std: 1.0 },
            ObservedValue::Scalar(1.5)
        );
        
        assert!(var.is_observed());
    }

    #[test]
    fn test_probabilistic_model() {
        let mut model = ProbabilisticModel::new("test_model".to_string());
        
        let var1 = RandomVariable::new(
            "mu".to_string(),
            Distribution::Normal { mean: 0.0, std: 1.0 }
        );
        
        let mut var2 = RandomVariable::new(
            "x".to_string(),
            Distribution::Normal { mean: 0.0, std: 1.0 }
        );
        var2.add_dependency("mu".to_string());
        
        model.add_random_variable(var1);
        model.add_random_variable(var2);
        
        assert!(model.validate().is_ok());
        assert_eq!(model.random_variables.len(), 2);
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        
        graph.add_edge("A".to_string(), "B".to_string());
        graph.add_edge("B".to_string(), "C".to_string());
        graph.update_topological_order();
        
        assert!(!graph.has_cycles());
        assert_eq!(graph.topological_order, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = DependencyGraph::new();
        
        graph.add_edge("A".to_string(), "B".to_string());
        graph.add_edge("B".to_string(), "C".to_string());
        graph.add_edge("C".to_string(), "A".to_string()); // Creates cycle
        
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig::mcmc(MCMCSampler::MetropolisHastings, 1000);
        
        assert_eq!(config.num_samples, 1000);
        assert_eq!(config.burn_in, 500);
        
        if let InferenceAlgorithm::MCMC { sampler, .. } = config.algorithm {
            assert_eq!(sampler, MCMCSampler::MetropolisHastings);
        } else {
            panic!("Expected MCMC algorithm");
        }
    }

    #[test]
    fn test_log_pdf() {
        let var = RandomVariable::new(
            "test".to_string(),
            Distribution::Normal { mean: 0.0, std: 1.0 }
        );
        
        let log_pdf_0 = var.log_pdf(0.0);
        let log_pdf_1 = var.log_pdf(1.0);
        
        // PDF should be higher at mean (0) than at 1 standard deviation away
        assert!(log_pdf_0 > log_pdf_1);
    }

    #[test]
    fn test_sampling() {
        let var = RandomVariable::new(
            "test".to_string(),
            Distribution::Uniform { min: 0.0, max: 1.0 }
        );
        
        let mut rng_state = 12345;
        let sample = var.sample(&mut rng_state);
        
        assert!(sample >= 0.0 && sample <= 1.0);
    }
}