// Perceptron implementation in Rust
use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use clap::{Arg, Command };
use ndarray::Array2;

//
fn main() {
    let cli = Command::new("Rusted Perceptron")
        .version("1.0")
        .author("Alejandro Nadal") 
        .about("Perceptron implementation in Rust")
        .arg(Arg::new("alpha")
            .short('a')
            .long("alpha")
            .value_name("ALPHA")
            .help("Sets the learning rate")
            .default_value("0.1")
        )
        .arg(Arg::new("epochs")
            .short('e')
            .long("epochs")
            .value_name("EPOCHS")
            .help("Sets the number of epochs")
            .default_value("100")
        )
        .arg(Arg::new("dataset")
            .short('d')
            .long("dataset")
            .value_name("DATASET")
            .help("Sets the dataset")
            .default_value("simple_dataset.csv")
        )
        .get_matches();
    // unwrap_or returns the value of alpha if it is defined, otherwise it returns 0.1
    let alpha = cli.get_one::<String>("alpha").unwrap().parse::<f32>().unwrap();
    let mut epochs = cli.get_one::<String>("epochs").unwrap().parse::<i32>().unwrap();
    let dataset = cli.get_one::<String>("dataset").unwrap();
    let contents = fs::read_to_string(dataset)
        .expect("Something went wrong reading the file");
    // Define an array with 13 elements
    // each element is a vector of 4 elements
    let mut dimensions = 0;
    let dataset = read_dataset(contents, &mut dimensions); 
    // we split the data in train_data, train_Y, test_data and test_Y
    let (mut train_data, train_y, mut test_data, test_y) = dataset_split(dataset);
    // we normalize the data
    min_max_norm(&mut train_data);
    min_max_norm(&mut test_data);
    // W is started randomly
    let mut w = randomly_initialize_weights(dimensions, None);
    let mut b = vec![0.0;dimensions as usize];
    println!("Starting weights {:?}", w);
    // train
    while epochs > 0{
        train(alpha, &mut w, &mut b, &train_data, &train_y, dimensions); 
        epochs -= 1;
    }
    //tesat
    println!("Weights at the end of training {:?}", w);
    let results = test(test_data, &test_y, &mut w, &mut b);
    let conf_matrix = confusion_matrix(results, test_y);
    println!("Confusion matrix {:?}", conf_matrix);
    let accuracy_val = accuracy(&conf_matrix);
    println!("Accuracy {:?}", accuracy_val);
}

fn min_max_norm(data: &mut Vec<Vec<f32>>){
    // first, we iterate over each column, finding the min and max for each column
    let mut min = vec![f32::INFINITY; data[0].len()];
    let mut max = vec![f32::NEG_INFINITY; data[0].len()];
    for i in 0..data.len(){
        for j in 0..data[0].len(){
            if data[i][j] < min[j]{
                min[j] = data[i][j];
            }
            if data[i][j] > max[j]{
                max[j] = data[i][j];
            }
        }
    }
    // second, we iterate over each column, normalizing each column
    for i in 0..data.len(){
        for j in 0..data[0].len(){
            data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
        }
    }
}
fn accuracy(conf_matrix : &Array2<i32>) -> f32{
    let mut total = 0;
    let mut correct = 0;
    let (rows, cols) = conf_matrix.dim();
    for i in 0..rows{
        for j in 0..cols{
            total += conf_matrix[[i,j]];
            if i == j{
                correct += conf_matrix[[i,j]];
            }
        }
    }
    correct as f32 / total as f32
}

fn train(alpha : f32, w : &mut Vec<f32>, b: &mut Vec<f32>, train_data : &Vec<Vec<f32>>, train_y: &Vec<f32>, dimensions : i32){
    let mut net :f32;
    let mut pred : f32;
    // iterate over the train data
    for(i, val) in train_data.iter().enumerate(){
        net = 0.0;
        // iterate over the dimensions
        for j in 0..dimensions{
            net += w[j as usize] * val[j as usize] + b[j as usize];
        }
        pred = step(net);
        // update the weights
        for j in 0..dimensions{
            w[j as usize] = w[j as usize] + alpha * (train_y[i] - pred) * val[j as usize];
            b[j as usize] = b[j as usize] + alpha * (train_y[i] - pred);
        }
    }
}



fn read_dataset(contents : String, dimensions : &mut i32) -> Vec<Vec<f32>>{
    // Converts the string file into a vector of floats 
    let mut dataset = Vec::new();
    // iterate over the lines of the file
    for (i, line) in contents.lines().enumerate() {
        // split the line into a vector of strings
        let v: Vec<&str> = line.split(",").collect();
        // convert the vector of strings into a vector of floats
        let v: Vec<f32> = v.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        dataset.push(v);
    }
    *dimensions = dataset[0].len() as i32 -1;
    dataset
}

fn test(test_data : Vec<Vec<f32>>, test_y: &Vec<f32>, w : &Vec<f32>, b: &Vec<f32>) -> Vec<f32>{
    let mut net :f32 = 0.0;
    let mut results = vec![0.0;test_data.len()]; 
    let dimensions = test_data[0].len() as i32;
    for(i, val) in test_data.iter().enumerate(){
        for j in 0..dimensions{
            net += w[j as usize] * val[j as usize] + b[j as usize];
        }
        results[i] = step(net);
    }
    println!("results = {:?}", results);
    println!("test_y = {:?}", test_y);
    results
}
fn dataset_split(mut dataset: Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let mut train_data : Vec<Vec<f32>> = vec![];
    let mut train_y = vec![];
    let mut test_data : Vec<Vec<f32>> = vec![];
    let mut test_y = vec![];

    // we shuffle the dataset
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng);

    // we split the dataset into train and test
    let pivot = ((dataset.len() as f64) * 0.8) as usize;

    for mut data_point in dataset {
        let y = data_point.pop().unwrap();

        if train_data.len() < pivot {
            train_data.push(data_point);
            train_y.push(y);
        } else {
            test_data.push(data_point);
            test_y.push(y);
        }
    }
    (train_data, train_y, test_data, test_y)
}



fn relu(val: f32) -> f32{
    if val > 0.0{
        return val
    } else{
        return 0.0
    }
}
fn step(val: f32) -> f32{
    if val > 0.0 {
        return 1.0
    } else {
        return -1.0 
    }
}

fn randomly_initialize_weights(dimensions: i32, seed: Option<u64>) -> Vec<f32> {
    // If a seed is provided, use it, otherwise use a thread local RNG
    let mut rng : Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng())
    };
    let mut w = Vec::new();
    for _i in 0..dimensions {
        w.push(rng.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0);
    }
    w
}

fn confusion_matrix(results: Vec<f32>, test_y: Vec<f32>) -> Array2<i32> {
    let mut conf_matrix = Array2::from_elem((2,2), 0);
    let dimensions = results.len();
    for i in 0..dimensions {
        if results[i] == 1.0 && test_y[i] == 1.0 {
            conf_matrix[[0,0]] +=1;
        } else if results[i] == 1.0 && test_y[i] == -1.0 {
            // this is a false positive
            conf_matrix[[0,1]] += 1;
        } else if results[i] == -1.0 && test_y[i] == 1.0 {
            // this is a false negative
            conf_matrix[[1,0]] += 1;
        } else {
            conf_matrix[[1,1]] += 1;
        }
    }
    conf_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_reproducibility() {
        let w1 = randomly_initialize_weights(10, Some(42));
        let w2 =  randomly_initialize_weights(10, Some(42));
        assert_eq!(w1, w2);
    }
}
