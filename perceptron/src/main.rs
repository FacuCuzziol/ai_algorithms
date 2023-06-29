// Perceptron implementation in Rust
use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use clap::{Arg, Command };

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
    println!("Text Input:\n{}", contents);
    // Define an array with 13 elements
    // each element is a vector of 4 elements
    let mut dimensions = 0;
    let dataset = read_dataset(contents, &mut dimensions); 
    // we split the data in train_data, train_Y, test_data and test_Y
    let (train_data, train_y, test_data, test_y) = dataset_split(dataset);
    // W is started randomly
    let mut w = randomly_initialize_weights(dimensions);
    let mut b = vec![0.0;dimensions as usize];
    println!("Starting weights {:?}", w);
    // train
    while epochs > 0{
        train(alpha, &mut w, &mut b, &train_data, &train_y, dimensions); 
        epochs -= 1;
    }
    //tesat
    println!("Weights at the end of training {:?}", w);
    test(test_data, test_y, &mut w, &mut b);
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
        pred = relu(net);
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
       println!("line = {}", line);
        // split the line into a vector of strings
        let v: Vec<&str> = line.split(",").collect();
        // convert the vector of strings into a vector of floats
        let v: Vec<f32> = v.iter().map(|x| x.parse::<f32>().unwrap()).collect();
        dataset.push(v);
    }
    *dimensions = dataset[0].len() as i32 -1;
    dataset
}

fn test(test_data : Vec<Vec<f32>>, test_y: Vec<f32>, w : &Vec<f32>, b: &Vec<f32>){
    let mut net :f32 = 0.0;
    let mut pred : f32 = 0.0;
    let mut results = vec![0.0;test_data.len()]; 
    for(i, val) in test_data.iter().enumerate(){
        for j in 0..3{
            net += w[j as usize] * val[j as usize] + b[j as usize];
        }
        pred = relu(net);
        if pred > 0.5{
            results[i] = 1.0;
        } else{
            results[i] = 0.0;
        }
    }
    println!("results = {:?}", results);
    println!("test_y = {:?}", test_y);
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

fn randomly_initialize_weights(dimensions: i32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut w = Vec::new();
    for _i in 0..dimensions {
        w.push(rng.gen_range(-1.0..1.0));
    }
    w
}
