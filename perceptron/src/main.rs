// Perceptron implementation in Rust
use rand::Rng;
use std::fs;
// 
fn main() {
    let contents = fs::read_to_string("simple_dataset.csv")
        .expect("Something went wrong reading the file");
    println!("With text:\n{}", contents);
    // Define an array with 13 elements
    // each element is a vector of 4 elements
    let dataset = read_dataset(contents); 
    println!("dataset elem 1 val 3 = {}", dataset[1][3]);
    let mut w = [0.0;3];
    let mut b = [0.0;3];
    let alpha = 0.1;
    let mut epochs = 100;
    // we split the data in train_data, train_Y, test_data and test_Y
    let (train_data, train_y, test_data, test_y) = dataset_split(dataset);
    println!("Train data {}",train_data[7][2]);
    // W is started randomly
    randomly_initialize_weights(&mut w);
    println!("w {:?}", w);
    // train
    while epochs > 0{
        train(alpha, &mut w, &mut b, train_data, train_y); 
        epochs -= 1;
    }
    //test
    println!("w {:?}", w);
    test(test_data, test_y, &mut w, &mut b);
}

fn train(alpha : f32, w : &mut [f32;3], b: &mut [f32;3], train_data : [[f32;3];8], train_y: [f32;8]){
    let mut net :f32 = 0.0;
    let mut pred : f32 = 0.0;
    for(i, val) in train_data.iter().enumerate(){
        net = net_input(val, w, b);
        pred = relu(net);
        w[0] = w[0] + alpha * (train_y[i] - pred) * val[0];
        w[1] = w[1] + alpha * (train_y[i] - pred) * val[1];
        w[2] = w[2] + alpha * (train_y[i] - pred) * val[2];
        b[0] = b[0] + alpha * (train_y[i] - pred);
        b[1] = b[1] + alpha * (train_y[i] - pred);
        b[2] = b[2] + alpha * (train_y[i] - pred);
    }
}



fn read_dataset(contents : String) -> [[f32;4];13]{
    let mut dataset = [[0.0;4];13];
    //print dataset
    println!("dataset = {:?}", dataset);
    // iterate over the lines of the file
    for (i, line) in contents.lines().enumerate() {
       println!("line = {}", line);
        // split the line into a vector of strings
        let v: Vec<&str> = line.split(",").collect();
        // iterate over the vector of strings
        for (j, s) in v.iter().enumerate() {
            // convert the string to a float
            let f: f32 = s.parse::<f32>().unwrap();
            // assign the float to the dataset
            dataset[i][j] = f;
        }
    }
    dataset
}

fn test(test_data : [[f32;3];5], test_y: [f32;5], w : &mut [f32;3], b: &mut [f32;3]){
    let mut net :f32 = 0.0;
    let mut pred : f32 = 0.0;
    let mut results = [0.0;5];
    for(i, val) in test_data.iter().enumerate(){
        net = net_input(val, w, b);
        pred = relu(net);
        if(pred > 0.5){
            results[i] = 1.0;
        } else{
            results[i] = 0.0;
        }
    }
    println!("results = {:?}", results);
    println!("test_y = {:?}", test_y);
}

fn dataset_split(dataset: [[f32;4];13] ) -> ([[f32;3];8], [f32;8], [[f32;3];5], [f32;5]){
    let mut train_data : [[f32;3];8] = [[0.0;3];8];
    let mut train_y = [0.0;8];
    let mut test_data = [[0.0;3];5];
    let mut test_y = [0.0;5];
    for (i, val) in dataset.iter().enumerate(){
        if i < 8{
            train_data[i].copy_from_slice(&val[0..3]);
            train_y[i] = val[3];
        } else{
            test_data[i-8].copy_from_slice(&val[0..3]);
            test_y[i-8] = val[3];
        }
    }
    (train_data, train_y, test_data, test_y)
}

fn net_input(data_row: &[f32;3], w: &mut [f32;3], b: &mut [f32;3]) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < 3{
        sum += w[i] * data_row[i] + b[i];
        i+=1;
    }
    sum
}

fn relu(val: f32) -> f32{
    if val > 0.0{
        return val
    } else{
        return 0.0
    }
}

fn randomly_initialize_weights(w: &mut [f32;3]) {
    w[0] = rand::thread_rng().gen_range(0.0..=1.0);
    w[1] = rand::thread_rng().gen_range(0.0..=1.0);
    w[2] = rand::thread_rng().gen_range(0.0..=1.0);
}
