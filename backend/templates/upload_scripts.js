function validate() {

         if( document.upload_form.patient_age.value < 0 ) {
            alert( "Invalid patient age" );
            document.myForm.Name.focus() ;
            return false;
         }
         alert("test!");
         return false;
         return( true );
      }