use std::{fs, env};
use std::path::{Path, PathBuf};
use std::error::Error;
use core::hash::{Hasher, BuildHasherDefault};
use std::collections::{HashMap, HashSet};

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

struct NaiveBayesProbabilities {
	spam: f64,
	legit: f64,
	word_spam: HashMap<u32, f64, BuildIdentityHasher>,
	word_legit: HashMap<u32, f64, BuildIdentityHasher>
}

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

fn file_to_array(filename: &PathBuf) -> Result<Vec<u32>, Box<dyn Error>> {
	let s = fs::read_to_string(filename)?;
	let ret: Vec<u32> = s.split_whitespace()
							.filter_map(|w| w.parse().ok())
							.collect();
    Ok(ret)
}

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
				//println!("{:?}", doc);
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
						*occurences_spam.entry(word).or_insert(0) += 1;
						n_spam += 1;
					}
					else {
						*occurences_legit.entry(word).or_insert(0) += 1;
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
		//println!("{}", word);
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

	//println!("{}, {}", p_spam, p_legit);
	//println!("{:?}", p_word_spam);
	//println!("{:?}", p_word_legit);

	NaiveBayesProbabilities{spam: p_spam, legit: p_legit, word_spam: p_word_spam, word_legit: p_word_legit}
}

fn classified_as_spam(filename: &PathBuf, p: &NaiveBayesProbabilities) -> Result<bool, Box<dyn Error>> {
	match file_to_array(filename) {
		Ok(doc) => {
			let mut spam = 0.0;
			let mut legit = 0.0;
			for word in doc {
				if let Some(x) = p.word_spam.get(&word) {
					//print!("{}\n", x);
					spam = spam + x.log10();
				}
				if let Some(x) = p.word_legit.get(&word) {
					//println!("{}", x);
					legit = legit + x.log10();
				}
			}

			//println!("spam:{}\nlegit:{}",spam, legit);
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
					_true_negative += 1;
				}
			}
			Err(e) => println!("{}", e)
		}
	}

	//println!("t+{}\nf+{}\nf-{}\nt-{}\ntotal{}", true_positive, false_positive, false_negative, true_negative, true_positive+false_positive+false_negative+true_negative);
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
						//println!("Spam recall: {}\nSpam precision: {}", r, p);
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

/*
	For doc in data_set{
		total ++ (=spam+legit)
		if spam {
			spam ++
		}
		else {
			legit ++
		}
		for word in doc {
			word_set.insert(word);
			if spam{
				occurences_spam[word]++
				n_spam++
			}
			if legit{
				occurences_legit[word]++
				n_legit++
			}
			total_words++
		}
	}

	P[spam] = spam / total
	P[legit] = legit / total

	for word in word_set{
		P_spam[word] = (occurences_spam[word] + 1) / (n_spam + total_words)
		P_legit[word] = (occurences_legit[word] + 1) / (n_legit + total_words)
	}
*/	