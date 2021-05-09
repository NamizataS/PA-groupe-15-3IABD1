use rand::Rng;


fn main() {
    let mut model = create_linear_model(3);
    let dataset_inputs = vec![1.0f64, 4.0, 1.0, -4.0, 4.0, 4.0];
    let dataset_expected_outputs = vec![1.0f64, 1.0, -1.0];
    println!("{:?}",model);
    for _ in 0..10{
        train_rosenblatt_linear_model(&mut model, &dataset_inputs, &dataset_expected_outputs, 20, 0.001);
    }
    println!("{:?}",model);
}

fn create_linear_model(x: i32) -> Vec<f64>{
    let mut rng = rand::thread_rng();
    let mut model = Vec::with_capacity(x as usize);
    for _ in 0..x{
        let mut num = rng.gen();
        num = num * 2.0 - 1.0;
        model.push(num);
    }
    model
}

fn predict_linear_model_regression(model: &Vec<f64>, inputs: &Vec<f64>) -> f64{
    let mut sum_rslt = model[0];
    for i in 1..model.len(){
        sum_rslt += model[i] * inputs[i-1];
    }
    sum_rslt
}

fn predict_linear_model_classification(model: &Vec<f64>, inputs: &Vec<f64>) -> f64{
    let pred = predict_linear_model_regression(&model,&inputs);
    return if pred >= 0.0 {1.0} else {-1.0}
}

fn train_rosenblatt_linear_model(model: &mut Vec<f64>, dataset_inputs: &Vec<f64>, dataset_expected_outputs: &Vec<f64>, iterations_count: i32, alpha: f64){
    let mut rng = rand::thread_rng();
    let input_usize = model.len() - 1;
    let sample_count = (dataset_inputs.len() / input_usize) as i32;
    let mut k = 0;
    let mut Xk;
    let mut yk = 0.0;
    let mut gXk = 0.0;
    for it in 0..iterations_count{
        k = rng.gen_range(0..sample_count) as usize;
        Xk = &dataset_inputs[k * input_usize..(k + 1) * input_usize];
        yk = dataset_expected_outputs[k];
        gXk = predict_linear_model_classification(model, &Xk.to_vec());

        model[0] += alpha * yk - gXk * 1.0;
        for i in 1..model.len(){
            model[i] += alpha * (yk - gXk) * Xk[i-1];
        }
    }
}
