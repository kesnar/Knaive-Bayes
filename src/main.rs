/// kesnar-naive Bayes (Knaive-Bayes) is a simple implementation of the Naive Bayes algorithm in Rust
/// Written by kesnar (Panagiotis Famelis) in December 2020
/// Published under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)

use std::{fs, env};
use std::path::{Path, PathBuf};
use std::error::Error;
use core::hash::{Hasher, BuildHasherDefault};
use std::collections::{HashMap, HashSet};

/// The identityHasher is the identity function f(x) = x.
/// The identity hash has been selected for speed reasons, as the Hash is used instead of a sparse Vector
#[derive(Debug, Clone, Copy, Default)]
struct IdentityHasher(usize);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }

    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("IdentityHasher only supports usize keys")
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = i as usize;
    }
}

type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

/// A struct for the collection of the probabilities needed for the Naive Bayes Classification.
struct NaiveBayesProbabilities {
	spam: f64,
	legit: f64,
	word_spam: HashMap<u32, f64, BuildIdentityHasher>,
	word_legit: HashMap<u32, f64, BuildIdentityHasher>
}

/// Recursive function to traverse the directorirs below a directory in a filesystem.
/// Returns error in case something went wrong.
fn visit_dirs(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
	let mut ret = Vec::new();
	if dir.is_dir() {
		for entry in fs::read_dir(dir)? {
			let entry = entry?;
			let path = entry.path();
			if path.is_dir() {
				ret.append(&mut visit_dirs(&path)?);
			} else {
				ret.push(path);
			}
		}
	}
	Ok(ret)
}

/// Function that opens and reads a file, returning a vector with the words in u32.
/// If there are words not in numeric format (i.e. "Subject:"), it discards them.
/// Returns error in case something went wrong.
fn file_to_array(filename: &PathBuf) -> Result<Vec<u32>, Box<dyn Error>> {
	let s = fs::read_to_string(filename)?;
	let ret: Vec<u32> = s.split_whitespace()
							.filter_map(|w| w.parse().ok())
							.collect();
    Ok(ret)
}

/// The function that calculates the Naive Bayes probabilities.
/// It takes as input a directory containing the data set, 
///  in format as described in http://www.aueb.gr/users/ion/data/PU123ACorpora.tar.gz
/// In case of error in some files, it disregards that file, prints and error message 
///  and continues to the next.
fn learn_naive_bayes(data_set: Vec<PathBuf>) -> NaiveBayesProbabilities {
	let mut total = 0;
	let mut spam = 0;
	let mut legit = 0;

	let mut n_spam = 0;
	let mut n_legit = 0;

	let mut word_set: HashSet<u32, BuildIdentityHasher> = HashSet::default();
	let mut occurences_spam: HashMap<u32, u32, BuildIdentityHasher> = HashMap::default();
	let mut occurences_legit: HashMap<u32, u32, BuildIdentityHasher> = HashMap::default();

	for doc in data_set.iter() {
		match file_to_array(doc) {
			Ok(example) => {
				let is_spam;
				total += 1;
				if doc.as_path().display().to_string().contains("spmsg") {
					spam += 1;
					is_spam = true;
				}
				else {//if doc.as_path().display().to_string().contains("legit") {
					legit += 1;
					is_spam = false;
				}
				for word in example {
					word_set.insert(word);
					if is_spam {
						// Increment the occurences by 1 or insert a new entry for the word with 1 occurence
						*occurences_spam.entry(word).or_insert(1) += 1;
						n_spam += 1;
					}
					else {
						// Increment the occurences by 1 or insert a new entry for the word with 1 occurence
						*occurences_legit.entry(word).or_insert(1) += 1;
						n_legit += 1;
					}
				}
			}
			Err(e) => println!("{}", e)
		}
	}

	let p_spam = spam as f64 / total as f64;
	let p_legit = legit as f64 / total as f64;

	let spam_divisor = (n_spam + word_set.len()) as f64;
	let legit_divisor= (n_legit + word_set.len()) as f64;
	
	let mut p_word_spam: HashMap<u32, f64, BuildIdentityHasher> = HashMap::default();
	let mut p_word_legit: HashMap<u32, f64, BuildIdentityHasher> = HashMap::default();
	
	for word in word_set {
		if let Some(x) = occurences_spam.get(&word) {
			p_word_spam.insert(word, (1 + x) as f64 / spam_divisor);
		} else {
			p_word_spam.insert(word, 1.0 / spam_divisor);
		}

		if let Some(x) = occurences_legit.get(&word) {
			p_word_legit.insert(word, (1 + x) as f64 / legit_divisor);
		} else {
			p_word_legit.insert(word, 1.0 / legit_divisor);
		}
	}

	NaiveBayesProbabilities{spam: p_spam, legit: p_legit, word_spam: p_word_spam, word_legit: p_word_legit}
}

/// Function that takes a filename and the Naive Bayes Probabilities and clasifies it as spam or not (boolean).
/// Returns error in case something went wrong.
/// As propabilities are < 1, multiplying them results on really small numbers close to zero, that are not
///  handled well. As such logarithms are used and the multiplication is trasformed to a sum of log10.
fn classified_as_spam(filename: &PathBuf, p: &NaiveBayesProbabilities) -> Result<bool, Box<dyn Error>> {
	match file_to_array(filename) {
		Ok(doc) => {
			let mut spam = 0.0;
			let mut legit = 0.0;
			for word in doc {
				if let Some(x) = p.word_spam.get(&word) {
					// Using log10 to acquire sum instead of multiplying. 
					spam = spam + x.log10();
				}
				if let Some(x) = p.word_legit.get(&word) {
					// Using log10 to acquire sum instead of multiplying.
					legit = legit + x.log10();
				}
			}

			// Using log10 to acquire sum instead of multiplying.
			if (spam + p.spam.log10()) >= (legit+p.legit.log10()) {
				Ok(true)
			} else {
				Ok(false)
			}
		}
		Err(e) => {
			Err(e)
		}
	}
}

/// Function that takes a directory and clasifies each mail in it as spam or legit.
/// Returns spam recall and spam precision.
fn test_naive_bayes(data_set: Vec<PathBuf>, p: &NaiveBayesProbabilities) -> (f64, f64) {
	let mut true_positive = 0;
	let mut false_positive = 0;
	let mut _true_negative = 0;
	let mut false_negative = 0;
	for doc in data_set.iter() {
		match classified_as_spam(doc, p) {
			Ok(true) => {
				if doc.as_path().display().to_string().contains("spmsg") {
					// Classified as spam and is spam
					true_positive += 1;
				}
				else {
					// Classified as spam and is legit
					false_positive += 1;
				}
			}
			Ok(false) => {
				if doc.as_path().display().to_string().contains("spmsg") {
					// Classified as legit and is spam
					false_negative += 1;
				}
				else{
					// Classified as legit and is legit
					// Variable is not used. Here for completness.
					_true_negative += 1;
				}
			}
			Err(e) => println!("{}", e)
		}
	}

	(true_positive as f64 / (true_positive + false_negative) as f64,
	 true_positive as f64 / (true_positive + false_positive) as f64)

}

fn main() {
	let args: Vec<String> = env::args().collect();

	if args.len() == 2 {
		let path = Path::new(&args[1]);
		if path.is_dir() {
			match visit_dirs(path) {
				Ok(filenames) => {
					let (_unused,used):(_,Vec<_>)=filenames.into_iter().partition(|x| x.as_path().display().to_string().contains("unused"));

					let mut recall = 0.0;
					let mut precision = 0.0;
					for i in 1..11 {
						println!("Now starting fold number {}", i);
						//MUST CHANGE TO REFLECT LINUX AND WINDOWS!!!!!!
						let (test,train):(_,Vec<_>)=used.clone().into_iter().partition(|x| x.as_path().display().to_string().contains(&format!("part{}\\",i)));
						let probabilities = learn_naive_bayes(train);
						let (r,p) = test_naive_bayes(test, &probabilities);
						recall += r;
						precision += p;
					}
					recall = recall / 10.0;
					precision = precision / 10.0;
					println!("Spam recall: {}\nSpam precision: {}", recall, precision);
				},
				Err(e) => println!("{}", e)
			}		
		}
		else {
			println!("Error: Directory not found!")
		}
	} else {
		println!("arg1: dataset directory");
	}
}