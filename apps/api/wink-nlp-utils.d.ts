// wink-nlp-utils.d.ts
declare module 'wink-nlp-utils' {
    interface ItsUtils {
      tokenize(): any;
      removeWords(): any;
      stem(): any;
      // Add other its functions as needed
    }
    
    interface StringUtils {
      lowerCase(): any;
      upperCase(): any;
      trim(): any;
      removeExtraSpaces(): any;
      // Add other string functions as needed
    }
    
    interface DefaultExport {
      its: ItsUtils;
      string: StringUtils;
      helper: any;
      tokens: any;
      // Add other modules as needed
    }
    
    const winkUtils: {
      default: DefaultExport;
    };
    
    export = winkUtils;
  }