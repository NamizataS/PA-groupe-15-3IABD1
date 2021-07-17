use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::iter::FromIterator;
use std::borrow::Borrow;
use nalgebra::{DMatrix, Matrix};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Array};
use libm::*;
use itertools::Itertools;
use osqp::{CscMatrix, Problem, Settings};
use std::array;


pub struct MLP{
    d: Vec<i32>,
    W: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>
}


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


#[no_mangle]
pub extern "C" fn train_model_SVM(dataset_inputs: *const f32, dataset_expected_outputs: *const f32, dataset_inputs_len: i32, dataset_expected_outputs_len: i32, dataset_inputs_dimension: i32) -> *mut f64 {


    let (dataset_inputs_slice, dataset_expected_outputs_slice) =
        unsafe {
            (from_raw_parts(dataset_inputs, dataset_inputs_len as usize),
             from_raw_parts(dataset_expected_outputs, dataset_expected_outputs_len as usize))
        };

    //Convertion of data_inputs from vec f32 to vec f64
    let mut dataset_inputs_vect_f64 = Vec::with_capacity(dataset_inputs_len as usize);

    for i in 0..dataset_inputs_len{
        dataset_inputs_vect_f64.push(f64::from(dataset_inputs_slice[i as usize]));
    }

    //Convertion of data_expected_outputs from vec f32 to vec f64
    let mut dataset_expected_outputs_vect_f64 = Vec::with_capacity(dataset_expected_outputs_len as usize);

    for i in 0..dataset_expected_outputs_len{
        dataset_expected_outputs_vect_f64.push(f64::from(dataset_expected_outputs_slice[i as usize]));
    }

    //Creation of the BigMatrix which is used to construct the matix P of OSQP.
    let mut BigMatrix:Vec<f64> = Vec::new();


    //Creation of the transpose of dataset_inputs
    let Xtranspose = DMatrix::from_iterator(dataset_inputs_dimension as usize, dataset_expected_outputs_len as usize, dataset_inputs_vect_f64.iter().cloned());

    for i in 0..dataset_expected_outputs_len {
        let Xl = DMatrix::from_iterator(1, dataset_inputs_dimension as usize, Xtranspose.column(i as usize).iter().cloned());
        for j in 0..dataset_expected_outputs_len {
            let Xc = DMatrix::from_iterator(1, dataset_inputs_dimension as usize, Xtranspose.column(j as usize).iter().cloned());
            let XtX = &Xl * &Xc.transpose();
            BigMatrix.push((dataset_expected_outputs_vect_f64[i as usize] * dataset_expected_outputs_vect_f64[j as usize] * &XtX.row(0)[0]));
        }
    }

    //Define problem data
    let mut P = Vec::new();

    let mut O =0 as usize;
    let mut E = dataset_expected_outputs_len as usize;
    for i in 0..dataset_expected_outputs_len {
        P.push(BigMatrix[O..E].to_vec());
        O = E;
        E+= dataset_expected_outputs_len as usize;
    }

    let mut A:Vec<Vec<f64>> = Vec::new();
    let mut tmp = Vec::new();

    A.push(dataset_expected_outputs_vect_f64.to_vec());

    for r in 0..dataset_expected_outputs_len {
        for i in 0..dataset_expected_outputs_len {
            if r == i {
                tmp.push(1.0);
            } else {
                tmp.push(0.0);
            }
        }
        A.push(tmp.to_vec());
        tmp.clear();
    }

    let mut q = Vec::with_capacity(dataset_expected_outputs_len as usize);

    for i in 0..dataset_expected_outputs_len{
        q.push(-1.0);
    }

    let mut l =Vec::with_capacity((dataset_expected_outputs_len+1) as usize);

    for i in 0..dataset_expected_outputs_len +1{
        l.push(0.0);
    }

    let mut u =Vec::with_capacity((dataset_expected_outputs_len+1) as usize);

    u.push(0.0);
    for i in 1..dataset_expected_outputs_len +1{
        u.push(f64::MAX);
    }

    let q = q.as_slice();
    let l = l.as_slice();
    let u = u.as_slice();


    // Extract the upper triangular elements of `P`
    let P = CscMatrix::from(&P).into_upper_tri();
    let A = CscMatrix::from(&A);

    // Disable verbose output
    let settings = Settings::default()
        .verbose(false);


    // Create an OSQP problem
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");

    // Solve problem
    let result = prob.solve();

    let alphas = result.x().unwrap();
    let mut alphasY = Vec::new();

    for i in 0..alphas.len(){
        alphasY.push(alphas[i]*dataset_expected_outputs_vect_f64[i]);
    }

    let mut W:Vec<f64> = Vec::new();

    for j in 0..dataset_inputs_dimension {
        W.push(Xtranspose.row(j as usize).iter().enumerate().map(|(i, x)| x*alphasY[i]).sum());
    }

    let mut sum : f64=0.0;
    let mut index= alphas.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();


    for i in 0..dataset_inputs_dimension {
        sum += W[i as usize]*dataset_inputs_vect_f64[(index* dataset_inputs_dimension as usize)+i as usize];
    }

    let W0 = (1.0/dataset_expected_outputs_vect_f64[index])-sum;

    W.insert(0,W0);

    W.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn radial_base_kernel(xn:*mut f64, xm:*mut f64,xn_len:usize,xm_len:usize) ->f64{

    let (xn_slice, xm_slice) =
        unsafe {
            (from_raw_parts(xn, xn_len as usize),
             from_raw_parts(xm, xm_len as usize))
        };


    let Xn = DMatrix::from_iterator( 1,xn_len as usize, xn_slice.iter().cloned());
    let Xm = DMatrix::from_iterator( 1,xm_len as usize, xm_slice.iter().cloned());


    let XnXn = &Xn*&Xn.transpose();
    let XmXm = &Xm*&Xm.transpose();
    let XnXm = &Xn*&Xm.transpose();

    exp(-1.0*(XnXn.row(0)[0]))*exp(-1.0*(XmXm.row(0)[0]))*exp(2.0*(XnXm.row(0)[0]))

}

pub extern "C" fn radial_base_kernel_trick(xn:*mut f64, xm:*mut f64,xn_len:usize,xm_len:usize) ->f64{

    let (xn_slice, xm_slice) =
        unsafe {
            (from_raw_parts(xn, xn_len as usize),
             from_raw_parts(xm, xm_len as usize))
        };

    (expf(-1.0 * (Array::from(xn_slice.to_vec()).dot(&Array::from(xn_slice.to_vec()))) as f32)
        * expf(-1.0 * (Array::from(xm_slice.to_vec()).dot(&Array::from(xm_slice.to_vec())))  as f32)
        * expf(2.0 * (Array::from(xn_slice.to_vec()).dot(&Array::from(xm_slice.to_vec()))) as f32)) as f64

}



#[no_mangle]
pub extern "C" fn train_model_SVM_trick(dataset_inputs: *const f32, dataset_expected_outputs: *const f32, dataset_inputs_len: i32, dataset_expected_outputs_len: i32, dataset_inputs_dimension: i32) -> *mut f64 {


    let (dataset_inputs_slice, dataset_expected_outputs_slice) =
        unsafe {
            (from_raw_parts(dataset_inputs, dataset_inputs_len as usize),
             from_raw_parts(dataset_expected_outputs, dataset_expected_outputs_len as usize))
        };

    //Convertion of data_inputs from vec f32 to vec f64
    let mut dataset_inputs_vect_f64 = Vec::with_capacity(dataset_inputs_len as usize);

    for i in 0..dataset_inputs_len{
        dataset_inputs_vect_f64.push(f64::from(dataset_inputs_slice[i as usize]));
    }

    //Convertion of data_expected_outputs from vec f32 to vec f64
    let mut dataset_expected_outputs_vect_f64 = Vec::with_capacity(dataset_expected_outputs_len as usize);

    for i in 0..dataset_expected_outputs_len{
        dataset_expected_outputs_vect_f64.push(f64::from(dataset_expected_outputs_slice[i as usize]));
    }

    //Creation of the BigMatrix which is used to construct the matix P of OSQP.
    let mut BigMatrix:Vec<f64> = Vec::new();


    //Creation of the transpose of dataset_inputs
    let Xtranspose = DMatrix::from_iterator(dataset_inputs_dimension as usize, dataset_expected_outputs_len as usize, dataset_inputs_vect_f64.iter().cloned());

    let mut Xn :Vec<f64>= Vec::with_capacity(dataset_inputs_dimension as usize);
    let mut Xm:Vec<f64>= Vec::with_capacity(dataset_inputs_dimension as usize);

    for i in 0..dataset_expected_outputs_len {
        for a in 0..dataset_inputs_dimension{
            Xn.push(Xtranspose.column(i as usize)[a as usize]);
        }
        for j in 0..dataset_expected_outputs_len {
            for a in 0..dataset_inputs_dimension{
                Xm.push(Xtranspose.column(j as usize)[a as usize]);
            }
            BigMatrix.push((dataset_expected_outputs_vect_f64[i as usize]* dataset_expected_outputs_vect_f64[j as usize]*radial_base_kernel(Xn.as_mut_ptr(),Xm.as_mut_ptr(),dataset_inputs_dimension as usize,dataset_inputs_dimension as usize)));
            Xm.clear();
        }
        Xn.clear();
    }

    //Define problem data
    let mut P = Vec::new();

    let mut O =0 as usize;
    let mut E = dataset_expected_outputs_len as usize;
    for i in 0..dataset_expected_outputs_len {
        P.push(BigMatrix[O..E].to_vec());
        O = E;
        E+= dataset_expected_outputs_len as usize;
    }

    let mut A:Vec<Vec<f64>> = Vec::new();
    let mut tmp = Vec::new();

    A.push(dataset_expected_outputs_vect_f64.to_vec());

    for r in 0..dataset_expected_outputs_len {
        for i in 0..dataset_expected_outputs_len {
            if r == i {
                tmp.push(1.0);
            } else {
                tmp.push(0.0);
            }
        }
        A.push(tmp.to_vec());
        tmp.clear();
    }

    let mut q = Vec::with_capacity(dataset_expected_outputs_len as usize);

    for i in 0..dataset_expected_outputs_len{
        q.push(-1.0);
    }

    let mut l =Vec::with_capacity((dataset_expected_outputs_len+1) as usize);

    for i in 0..dataset_expected_outputs_len +1{
        l.push(0.0);
    }

    let mut u =Vec::with_capacity((dataset_expected_outputs_len+1) as usize);

    u.push(0.0);
    for i in 1..dataset_expected_outputs_len +1{
        u.push(f64::MAX);
    }

    let q = q.as_slice();
    let l = l.as_slice();
    let u = u.as_slice();


    // Extract the upper triangular elements of `P`
    let P = CscMatrix::from(&P).into_upper_tri();
    let A = CscMatrix::from(&A);

    // Disable verbose output
    let settings = Settings::default()
        .verbose(false);


    // Create an OSQP problem
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");

    // Solve problem
    let result = prob.solve();

    let alphas = result.x().unwrap();
    let mut alphasY = Vec::new();

    for i in 0..alphas.len(){
        alphasY.push(alphas[i]*dataset_expected_outputs_vect_f64[i]);
    }

    let mut sum : f64=0.0;

    let mut index= alphas.iter().position_max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let mut Xk= Vec::with_capacity(dataset_inputs_dimension as usize);
    let mut XSavPov =  Vec::with_capacity(dataset_inputs_dimension as usize);

    for a in 0..dataset_inputs_dimension{
        XSavPov.push(Xtranspose.column(index as usize)[a as usize]);
    }

    for i in 0..dataset_expected_outputs_len {
        //sum += W[i]* dataset_inputs[(index* dataset_inputs_dimension)+i];
        for a in 0..dataset_inputs_dimension{
            Xk.push(Xtranspose.column(i as usize)[a as usize]);
        }
        sum+=alphas[i as usize]*dataset_expected_outputs_vect_f64[i as usize]*radial_base_kernel(Xk.as_mut_ptr(),XSavPov.as_mut_ptr(),dataset_inputs_dimension as usize,dataset_inputs_dimension as usize);
        Xk.clear();
    }

    let W0 = (1.0/ dataset_expected_outputs_vect_f64[index])-sum;

    let mut W:Vec<f64> = Vec::new();

    let mut Xc = Vec::new();
    let mut sumW = 0.0;

    let mut dataset_inputs_vec = Vec::new();
    for i in 0..dataset_inputs_len{
        dataset_inputs_vec.push(dataset_inputs_vect_f64[i as usize] as f64);
    }

    for j in 0..dataset_expected_outputs_len {
        for a in 0..dataset_inputs_dimension {
            Xc.push(Xtranspose.column(j as usize)[a as usize]);
        }

        sumW += alphas[j as usize]*dataset_expected_outputs_vect_f64[j as usize]*radial_base_kernel_trick(Xc.as_mut_ptr(),dataset_inputs_vect_f64.as_mut_ptr(),dataset_inputs_dimension as usize,dataset_inputs_len as usize);
        Xc.clear();
    }

    W.push((sumW+W0));

    W.insert(0,W0);

    W.as_mut_ptr()
}




#[no_mangle]
pub extern "C" fn predict_SVM(model: *mut f64,dataset_inputs: *const f64, model_len:i32) -> f32 {


    let (model_slice, dataset_inputs_slice) =
        unsafe {
            (from_raw_parts_mut(model, model_len as usize),
             from_raw_parts(dataset_inputs, model_len as usize))
        };

    let model_matrix = DMatrix::from_iterator( 1,model_len as usize, model_slice.iter().cloned());
    let dataset_inputs_matrix = DMatrix::from_iterator( model_len as usize,1, dataset_inputs_slice.iter().cloned());

    let mut pred = &model_matrix*&dataset_inputs_matrix;
    let mut res:f32;

    if pred.row(0)[0]>=0.0{
        res = 1.0 as f32;
    }else{
        res = -1.0 as f32;
    }

    res

}


#[no_mangle]
pub extern "C" fn destroy_array(arr: *mut f32, arr_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(arr, arr_size as usize, arr_size as usize);
    }
}

#[no_mangle]
pub extern "C" fn destroy_array_double(arr: *mut f64, arr_size: i32) {
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


