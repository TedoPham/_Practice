function convertToRoman(num) {
    function getSymbol(n){
        switch(n){
            case 1: return "I";
            case 4: return "IV";  
            case 5: return "V";
            case 9: return "IX";
            case 10: return "X";
            case 40: return "XL";
            case 50: return "L";
            case 90: return "XC"
            case 100: return "C";
            case 400: return "CD";
            case 500: return "D";
            case 900: return "CM"
            case 1000: return "M";
        }
    }
    /*
    console.log(getSymbol(5));
    console.log(typeof("123".repeat(3)));
    console.log(typeof("123".concat("456")));
    */
    let x = [1,4,5,9,10,40,50,90,100,400,500,900,1000];
    let index = x.length - 1;
    let retStr = "";
    while(num > 0){
        let rep = Math.floor(num/x[index]);    
        if(rep >= 1){
            num -= rep*x[index];
            //console.log(getSymbol(x[i]).repeat(rep));
            retStr += getSymbol(x[index]).repeat(rep);  
        }
        //console.log(rep, x[i], num);
        index--;
    }

    console.log(retStr);
    return retStr;
}

convertToRoman(360);
convertToRoman(1004);