
pub const DNA_ALPH: [bool; 128] = {
    let mut table = [false; 128];
    let chars = b"ACGTNRYSWKMBDHVacgtnryswkmbdhv";
    let mut i = 0;
    while i < chars.len() {
        table[chars[i] as usize] = true;
        i += 1;
    }
    table
};

pub const RNA_ALPH: [bool; 128] = {
    let mut table = [false; 128];
    let chars = b"ACGUNRYSWKMBDHVacgunryswkmbdhv";
    let mut i = 0;
    while i < chars.len() {
        table[chars[i] as usize] = true;
        i += 1;
    }
    table
};

/// 20 + ambiguity (B, Z, X) + stop (*) + gap (-)
pub const PROTEIN_ALPH: [bool; 128] = { let mut table = [false; 128];
    let chars = b"ACDEFGHIKLMNPQRSTVWYBXZacdefghiklmnpqrstvwybxz*-";
    let mut i = 0;
    while i < chars.len() {
        table[chars[i] as usize] = true;
        i += 1;
    }
    table
};

pub fn is_dna(char: u8) -> bool {
    return char < 128 && DNA_ALPH[char as usize];
}

pub fn is_rna(char: u8) -> bool {
    return char < 128 && RNA_ALPH[char as usize];
}

pub fn is_prot(char: u8) -> bool {
    return char < 128 && PROTEIN_ALPH[char as usize];
}
