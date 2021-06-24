use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::iter::FromIterator;
use std::borrow::Borrow;
use nalgebra::{DMatrix, Matrix};
use libm::*;


pub struct MLP{
    d: Vec<i32>,
    W: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>
}

pub struct RBF{
    W: Vec<f32>,
    mu: Vec<f32>,
    gamma: Vec<f32>,
    K: i32,
    sample_count: usize,
    input_dim: usize
}

//LINEAR MODEL
#[no_mangle]
pub extern "C" fn create_model(x: i32) -> *mut f32{
    let mut rng = rand::thread_rng();
    let mut model = Vec::with_capacity(x as usize);
    for _ in 0..(x+1){
        let mut num = rng.gen_range(-1.0..1.0); //1.0 not include
        if num>1.0{
            num=1.0;
        }
        model.push(num);
    }
    let boxed_slice = model.into_boxed_slice();
    let model_ref = Box::leak(boxed_slice);
    model_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *const f32, inputs: *const f32, model_size: i32, inputs_size: i32) -> f32{
    let model = unsafe{
        from_raw_parts(model, model_size as usize)
    };
    let inputs = unsafe{
        from_raw_parts(inputs,inputs_size as usize)
    };
    let mut sum_rslt = model[0];
    for i in 1..model.len(){
        sum_rslt += model[i] * inputs[i-1];
    }
    sum_rslt
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *const f32, inputs: *const f32, model_size: i32, input_size: i32) -> f32{
    let pred = predict_linear_model_regression(model,inputs, model_size, input_size);
    let rslt;
    if pred >= 0.0{
        rslt = 1.0;
    } else{
        rslt = -1.0;
    }
    rslt as f32

}

#[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(model: *mut f32, dataset_inputs: *const f32, dataset_expected_outputs: *const f32, iterations_count: i32, alpha: f32, model_len: i32, inputs_len: i32, outputs_len: i32){
    let mut rng = rand::thread_rng();
    let mut model = unsafe{
        from_raw_parts_mut(model, model_len as usize)
    };
    let dataset_inputs = unsafe{
        from_raw_parts(dataset_inputs, inputs_len as usize)
    };
    let dataset_expected_outputs = unsafe{
        from_raw_parts(dataset_expected_outputs, outputs_len as usize)
    };
    let model_len_usize = (model_len - 1) as usize;
    let sample_count = inputs_len / (model_len - 1);
    let mut k = 0;
    let mut Xk;
    let mut yk = 0.0;
    let mut gXk = 0.0 as f32;
    for it in 0..iterations_count{
        k = rng.gen_range(0..sample_count) as usize;
        Xk = &dataset_inputs[k * model_len_usize..(k + 1) * model_len_usize];
        yk = dataset_expected_outputs[k];
        gXk = predict_linear_model_classification(model.as_ptr(), Xk.as_ptr(), model_len, inputs_len );
        model[0] += alpha * (yk - gXk) * 1.0;
        for i in 1..(model_len_usize + 1){
            model[i] += alpha * (yk - gXk) * Xk[i-1];
        }
    }
}

#[no_mangle]
pub extern "C" fn train_regression_linear_model(model: *mut f32, dataset_inputs: *const f32, dataset_expected_outputs: *const f32, model_size: i32, dataset_inputs_len: i32, dataset_outputs_len: i32, input_dim: i32, output_dim: i32){

    let (dataset_inputs_slice, dataset_outputs_slice) =
        unsafe {
            (from_raw_parts(dataset_inputs, dataset_inputs_len as usize),
             from_raw_parts(dataset_expected_outputs, dataset_outputs_len as usize))
        };

    let mut model = unsafe{
        from_raw_parts_mut(model, model_size as usize)
    };

    let mut model_dim = (model_size - 1) as usize;
    let mut sample_count = (dataset_inputs_len as usize) / model_dim;

    let X = DMatrix::from_iterator(sample_count, input_dim as usize, dataset_inputs_slice.iter().cloned());
    let Y = DMatrix::from_iterator(sample_count, output_dim as usize, dataset_outputs_slice.iter().cloned());


    let X = X.insert_columns(0, 1, 1.0);

    let XtX = &X.transpose() * &X;

    let XtXInv = XtX.cholesky().unwrap().inverse();

    let W = (XtXInv * &X.transpose()) * Y;

    for (i, row) in W.row_iter().enumerate() {
        model[i] =W.row(i)[0];
    }
}

//MLP MODEL

#[no_mangle]
pub extern "C" fn to_vec_int(tmp_vec : &[i32])->Vec<i32>{
    let mut d: Vec<i32> = Vec::with_capacity(tmp_vec.len());
    for i in 0..tmp_vec.len(){
        d.push(tmp_vec[i]);
    }
    d
}

#[no_mangle]
pub extern "C" fn to_vec_float(tmp_vec: &[f32])->Vec<f32>{
    let mut d: Vec<f32> = Vec::with_capacity(tmp_vec.len());
    for i in 0..tmp_vec.len(){
        d.push(tmp_vec[i]);
    }
    d
}

#[no_mangle]
pub extern "C" fn create_mlp_model(npl: *mut i32, npl_len: i32) -> *mut MLP{
    let mut rng = rand::thread_rng();
    let uni = Uniform::from(-1.0..1.0);
    let mut W : Vec<Vec<Vec<f32>>> = Vec::new();
    let mut tmp_d = unsafe {
        from_raw_parts(npl, npl_len as usize)
    };
    let mut d = to_vec_int(tmp_d);

    for l in 0..npl_len{ //for each layer
        let mut new_vec_w: Vec<Vec<f32>> = Vec::new();
        let mut l_usize = l as usize;
        if l== 0 { //no previous layer so empty
            W.push(new_vec_w);
            continue;
        }
        for i in 0..(d[(l_usize-1)]+1){ //for each neuron of the previous layer/l-1 represent the previous layer
            let mut new_vec: Vec<f32> = Vec::new();
            for j in 0..(d[l_usize]+1){ //for each neuron of the next layer
                let mut num = uni.sample(&mut rng) as f32;
                new_vec.push(num);
            }
            new_vec_w.push(new_vec);
        }
        W.push(new_vec_w);
    }
    let mut X : Vec<Vec<f32>> = Vec::new(); //output of each neuron
    for l in 0..npl_len{
        let mut l_usize = l as usize;
        let mut new_vec_X: Vec<f32> = Vec::new();
        for j in 0..(d[l_usize]+1){
            let mut j_usize = j as usize;
            let mut num;
            if j == 0{
                num = 1.0;
            } else {
                num = 0.0;
            }
            new_vec_X.push(num);
        }
        X.push(new_vec_X);
    }
    let mut deltas : Vec<Vec<f32>> = Vec::new();
    for l in 0..npl_len{
        let mut l_usize = l as usize;
        let mut new_vec_deltas : Vec<f32> = Vec::new();
        for j in 0..(d[l_usize]+1){
            let mut j_usize = j as usize;
            new_vec_deltas.push(0.0);
        }
        deltas.push(new_vec_deltas);
    }
    let model = MLP {
        d,
        W,
        x: X,
        deltas
    };
    let model_leaked = Box::leak(Box::from(model));
    model_leaked as *mut MLP
}

#[no_mangle]
pub extern "C" fn forward_pass(model: &mut MLP, sample_inputs: *const f32, is_classification: bool, inputs_len: i32){//compute outputs of every neuron of every layer
    let sample_inputs_tmp = unsafe{
        from_raw_parts(sample_inputs, inputs_len as usize)
    };
    let mut sample_inputs = to_vec_float(sample_inputs_tmp);

    for j in 1..(model.d[0]+1){
        let mut j_usize = j as usize;
        model.x[0][j_usize] = sample_inputs[j_usize-1];
    }
    for l in 1..model.d.len() {
        for j in 1..model.d[l]+1{
            let mut j_usize = j as usize;
            let mut sum_result = 0.0;
            for i in 0..(model.d[l - 1] + 1){
                let mut i_usize = i as usize;
                sum_result += model.W[l][i_usize][j_usize] * model.x[l-1][i_usize];
            }
            model.x[l][j_usize] = sum_result;
            if l < (model.d.len() - 1) || is_classification {
                model.x[l][j_usize] = tanhf(model.x[l][j_usize]);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn train_stochastic_gradient_backpropagation(model: &mut MLP, flattened_dataset_inputs: *mut f32, flatenned_expected_outputs: *mut f32, is_classification: bool, alpha: f32, iteration_count: i32, inputs_len: i32, output_len: i32){
    let mut rng = rand::thread_rng();
    let mut L = model.d.len() - 1;
    let input_dim = model.d[0] as usize;
    let output_dim = model.d[L] as usize;
    let sample_count = inputs_len / input_dim as i32;

    let mut flattened_dataset_inputs = unsafe{
        from_raw_parts(flattened_dataset_inputs, inputs_len as usize)
    };
    let mut flattened_expected_outputs = unsafe{
        from_raw_parts(flatenned_expected_outputs, output_len as usize)
    };
    for it in 0..iteration_count{
        let mut k = (rng.gen_range(0..sample_count)) as usize;
        let mut inputs_len = (k+1)*input_dim - k*input_dim;
        let mut sample_inputs = &flattened_dataset_inputs[k*input_dim..(k+1)*input_dim];
        let mut sample_expected_outputs = &flattened_expected_outputs[k*output_dim..(k+1)*output_dim];
        forward_pass(model,sample_inputs.as_ptr(),is_classification,inputs_len as i32);
        for j in 1..(model.d[L] as usize+1){
            model.deltas[L][j] = model.x[L][j] - sample_expected_outputs[j-1];
            if is_classification{
                model.deltas[L][j] = (1.0 - model.x[L][j] * model.x[L][j] )*model.deltas[L][j];
            }
        }
        for l in (1..(L+1)).rev(){
            for i in 0..(model.d[l-1]+1) as usize{
                let mut sum_result = 0.0;
                for j in 1..(model.d[l]+1) as usize{
                    sum_result += model.W[l][i][j] * model.deltas[l][j];
                }
                model.deltas[l-1][i] = (1.0 - model.x[l-1][i] * model.x[l-1][i]) * sum_result;
            }
        }
        for l in 1..(L+1){
            for i in 0..(model.d[l-1]+1) as usize{
                for j in 1..(model.d[l] +1) as usize{
                    model.W[l][i][j] += - alpha * model.x[l-1][i] * model.deltas[l][j];
                }
            }
        }
    }

}

#[no_mangle]
pub extern "C" fn predict_mlp_model_classification(model: *mut MLP, sample_inputs: *mut f32, inputs_len:i32) -> *mut f32 {
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    forward_pass(model,sample_inputs,true,inputs_len);
    let L = (model.d.len() - 1) as usize;
    let i = (model.d[L] + 1) as usize;
    let mut result:&mut[f32] = &mut model.x[L][1..i];
    result.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_classification_stochastic_backdrop_mlp_model(model: *mut MLP, flattened_dataset_inputs: *mut f32, flattened_expected_outputs: *mut f32, alpha: f32, iterations_count: i32, inputs_len: i32, outputs_len: i32){
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    train_stochastic_gradient_backpropagation(model,flattened_dataset_inputs,flattened_expected_outputs, true, alpha, iterations_count, inputs_len, outputs_len);
}

#[no_mangle]
pub extern "C" fn train_regression_stochastic_backdrop_mlp_model(model: *mut MLP, flattened_dataset_inputs: *mut f32, flattened_expected_outputs: *mut f32, alpha: f32, iterations_count: i32, inputs_len: i32, outputs_len: i32){
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    train_stochastic_gradient_backpropagation(model,flattened_dataset_inputs,flattened_expected_outputs, false, alpha, iterations_count, inputs_len, outputs_len);
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_regression(model: *mut MLP, sample_inputs: *mut f32, inputs_len:i32) -> *mut f32 {
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    forward_pass(model,sample_inputs,false,inputs_len);
    let L = (model.d.len() - 1) as usize;
    let i = (model.d[L] + 1) as usize;
    let mut result:&mut[f32] = &mut model.x[L][1..i];
    result.as_mut_ptr()
}

//RBF

#[no_mangle]
pub extern "C" fn create_rbf_model(K: i32, dataset_inputs: *mut f32, dataset_inputs_len: i32, input_dim: i32) -> *mut RBF{
    let mut rng = rand::thread_rng();
    let mut W:Vec<f32> = Vec::new();
    let mut gamma:Vec<f32> = Vec::new();
    let mut mu:Vec<f32> = Vec::new();
    let dataset_inputs = unsafe{
        from_raw_parts(dataset_inputs,dataset_inputs_len as usize)
    };
    let sample_count = (dataset_inputs_len / input_dim) as usize;
    let input_dim = input_dim as usize;
    for _ in 0..K{
        let mut num = rng.gen_range(-1.0..1.0);
        W.push(num);
    }
    let mut k_vec = Vec::new();
    for count in 0..K{
        let mut k = rng.gen_range(0..(sample_count));
        if count != 0 {
            while k_vec.contains(&k){
                k = rng.gen_range(0..sample_count);
            }
        }
        k_vec.push(k);
        let mut num = &dataset_inputs[k * (input_dim)..(k+1) * (input_dim)];
        for i in 0..num.len(){
            mu.push(num[i]);
        }
    }

    for _ in 0..K{
        let mut num = rng.gen_range(0.0..1.0);
        gamma.push(num);
    }

    let model = RBF{
        W,
        mu,
        gamma,
        K,
        sample_count,
        input_dim
    };
    let model_leaked = Box::leak(Box::from(model));
    model_leaked as *mut RBF
}

#[no_mangle]
pub extern "C" fn train_rbf_model(model: &mut RBF, flattened_dataset_inputs: *mut f32, dataset_outputs: *mut f32, dataset_inputs_len: i32, dataset_output_len: i32, output_dim: i32){
    let mut phi:Vec<f32> = Vec::new();
    let flattened_dataset_inputs = unsafe{
        from_raw_parts(flattened_dataset_inputs, dataset_inputs_len as usize)
    };
    let dataset_outputs = unsafe{
        from_raw_parts(dataset_outputs, dataset_output_len as usize)
    };
    for mut i in (0..(dataset_inputs_len as usize)).step_by(model.input_dim){
        let mut sample_inputs = &flattened_dataset_inputs[i..i+(model.input_dim)];
        let mut gamma_count = 0 as usize;
        for mut j in (0..(model.mu.len())).step_by(model.input_dim){
            let mut distance = 0.0f32;
            let mut sample_mu = &model.mu[j..j+model.input_dim];
            for k in 0..model.input_dim{
                distance += powf((sample_inputs[k]-sample_mu[k]), 2.0);
            }
            distance = sqrtf(distance);
            distance = powf(distance, 2.0);
            let mut value = expf(-model.gamma[gamma_count] * distance);
            phi.push(value);
            gamma_count += 1;
        }
    }
    let phi = DMatrix::from_iterator(model.sample_count, model.K as usize, phi.iter().cloned());
    let Y = DMatrix::from_iterator(model.sample_count, output_dim as usize, dataset_outputs.iter().cloned());
    let phiTphi = &phi.transpose() * &phi;
    let phiTphiInv = phiTphi.cholesky().unwrap().inverse();
    let W = (phiTphiInv * &phi.transpose()) * Y;
    for (i, row) in W.row_iter().enumerate() {
        model.W[i] =W.row(i)[0];
    }
}

#[no_mangle]
pub extern "C" fn predict_rbf_model_regression(model: *mut RBF, sample_inputs: *const f32)->f32{
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    let sample_inputs = unsafe{
        from_raw_parts(sample_inputs, model.input_dim)
    };
    let mut sum_result = 0.0f32;
    for i in 0..model.W.len(){
        let mut distance = 0.0f32;
        for j in 0..model.input_dim{
            distance += powf(sample_inputs[j] - model.mu[i+j], 2.0);
        }
        distance = sqrtf(distance);
        distance = powf(distance, 2.0);
        sum_result += model.W[i] * expf(-model.gamma[i] *distance );
    }
    sum_result
}

#[no_mangle]
pub extern "C" fn predict_rbf_model_classification(model: *mut RBF, sample_inputs: *const f32)->f32{
    let pred = predict_rbf_model_regression(model, sample_inputs);
    let mut rslt = 0.0f32;
    if pred >= 0.0{
        rslt = 1.0;
    } else{
        rslt = -1.0;
    }
    rslt
}
#[no_mangle]
pub extern "C" fn train_em_rbf_model_classification(model: *mut RBF, flattened_dataset_inputs: *mut f32, dataset_outputs: *mut f32, dataset_inputs_len: i32, dataset_outputs_len: i32, output_dim: i32, learning_rate: f32, iterations: i32){
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    train_em_rbf_model(model, flattened_dataset_inputs, dataset_outputs, dataset_inputs_len, dataset_outputs_len,output_dim, learning_rate, iterations, true);
}
#[no_mangle]
pub extern "C" fn train_em_rbf_model_regression(model: *mut RBF, flattened_dataset_inputs: *mut f32, dataset_outputs: *mut f32, dataset_inputs_len: i32, dataset_outputs_len: i32, output_dim: i32, learning_rate: f32, iterations: i32){
    let mut model = unsafe{
        model.as_mut().unwrap()
    };
    train_em_rbf_model(model, flattened_dataset_inputs, dataset_outputs, dataset_inputs_len, dataset_outputs_len,output_dim, learning_rate, iterations, false);
}
#[no_mangle]
pub extern "C" fn train_em_rbf_model(model: &mut RBF, flattened_dataset_inputs: *mut f32, dataset_outputs: *mut f32, dataset_inputs_len: i32, dataset_outputs_len: i32, output_dim: i32, learning_rate: f32, iterations: i32, is_classification: bool){
    for i in 0..(model.K as usize){
        let precision = 0.000001;
        let mut previous_step_rate = 1.0f32;
        let mut iterations_count = 0;
        while previous_step_rate > precision && iterations_count < iterations{
            train_rbf_model(model, flattened_dataset_inputs, dataset_outputs, dataset_inputs_len, dataset_outputs_len, output_dim);
            let mut prev_gamma = model.gamma[i];
            model.gamma[i] -= learning_rate * gradient(model, i, flattened_dataset_inputs, dataset_inputs_len, dataset_outputs, dataset_outputs_len, is_classification);
            previous_step_rate = model.gamma[i] - prev_gamma;
            iterations_count += 1;
        }
    }
}

#[no_mangle]
pub extern "C" fn gradient(model: &mut RBF, i: usize, flattened_dataset_inputs: *mut f32, inputs_len: i32, dataset_outputs: *mut f32, outputs_len: i32,is_classification: bool)->f32{
    let mut flattened_dataset_inputs = unsafe{
        from_raw_parts(flattened_dataset_inputs, inputs_len as usize)
    };
    let mut dataset_outputs = unsafe{
        from_raw_parts(dataset_outputs, outputs_len as usize)
    };
    let mut sum_result = 0.0f32;
    for j in 0..model.sample_count{
        let mut sample_inputs = &flattened_dataset_inputs[j..(j+model.input_dim)];
        let mut sample_mu = &model.mu[i..(i+model.input_dim)];
        let mut distance = 0.0f32;
        for k in 0..model.input_dim{
            distance += powf(sample_inputs[k] - sample_mu[k], 2.0);
        }
        distance = sqrtf(distance);
        let mut pred = 0.0f32;
        if is_classification{
            pred = predict_rbf_model_classification(model, sample_inputs.as_ptr());
        } else {
            pred = predict_rbf_model_regression(model, sample_inputs.as_ptr());
        }
        sum_result += 2.0 * distance * ( dataset_outputs[j] - pred ) * pred;
    }
    sum_result
}

#[no_mangle]
pub extern "C" fn destroy_array(arr: *mut f32, arr_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(arr, arr_size as usize, arr_size as usize);
    }
}

#[no_mangle]
pub extern "C" fn destroy_model(model: *mut MLP){
    unsafe{
        let _ = Box::from_raw(model);
    }
}

#[no_mangle]
pub extern "C" fn destroy_rbf_model(model: *mut RBF){
    unsafe{
        let _ = Box::from_raw(model);
    }
}