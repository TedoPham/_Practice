function CaesarsCipher(str, i) {
  let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  let offset = i;
  let str2 = [...str].map( (item)=> {
    if(/\w/.test(item)){
      let index = (alphabet.indexOf(item)+offset) %26;
      item = alphabet.substring(index,index+1);
    }
    return item;
  });
  return str2.join('');
}

console.log(CaesarsCipher("SERR PBQR PNZC",13));