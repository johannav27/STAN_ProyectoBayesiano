
data {
  //data size
  int<lower=0> N; // # de datos
  int<lower=0> C; //# de covariables
  //Datos Covariables
  matrix[N,C] X_train; // training data
  //Datos resultado de partidos
  int<lower=0,upper=1> match_outcome[N];
  int<lower=0> N_test;
  matrix[N_test,C] X_test;
}

parameters {
  vector[C] coeffs;
  real<lower=0> intercept;
}

transformed parameters {
  vector[N] y_prob;
  y_prob = inv_logit(X_train*coeffs + intercept);
}

model {
  //Priori
  intercept ~ normal(3,3);
  coeffs[1] ~ normal(8,2);
  coeffs[2] ~ normal(5,4);
  coeffs[3] ~ normal(5,4);
  coeffs[4] ~ normal(0,.1);

  //Verosimilitud
   match_outcome ~ bernoulli(y_prob);

}

generated quantities{
  real y_pred_test[N_test];
  for (i in 1:N_test){
    y_pred_test[i] = inv_logit(X_test[i,]*coeffs + intercept);
  }

}
