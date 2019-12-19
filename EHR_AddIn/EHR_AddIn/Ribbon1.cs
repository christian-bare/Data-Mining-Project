using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Office.Tools.Ribbon;
using Excel = Microsoft.Office.Interop.Excel;
using Office = Microsoft.Office.Core;
using Microsoft.Office.Tools.Excel;
using System.Windows.Forms;

namespace EHR_AddIn
{
    public partial class Ribbon1
    {
        private void Ribbon1_Load(object sender, RibbonUIEventArgs e)
        {

        }

        private void Decrypt_Click(object sender, RibbonControlEventArgs e)
        {

            Excel.Worksheet currentWorksheet = Globals.ThisAddIn.Application.ActiveSheet;
            MessageBox.Show("Hello");
            currentWorksheet.Range["A1"].Value = "HelloWorld";
            bool flag = AuthUser();

            if (flag == true)
            {
                MessageBox.Show("Welcome");
            }

            else
            {
                MessageBox.Show("Didn't work.");
            }
        }

        public bool AuthUser()
        {

            int i = 1;
            bool flag = false;
            while (i < 4)
            {
                string user = Globals.ThisAddIn.Application.InputBox("Enter your Username: ");
                string pass = Globals.ThisAddIn.Application.InputBox("Enter your Password: ");

                if (user == "admin")
                {
                    i++;
                    if (pass == "admin")
                    {
                        i = 4;
                        flag = true;
                    }
                }

                else if (user == "jdoe")
                {
                    i++;
                    if (pass == "jdoe")
                    {
                        i = 4;
                        flag = true;
                    }
                }

                else if (user == "cclark")
                {
                    i++;
                    if (pass == "cclark")
                    {
                        i = 4;
                        flag = true;
                    }
                }

                else if (user == "bsmith")
                {
                    i++;
                    if (pass == "bsmith")
                    {
                        i = 4;
                        flag = true;
                    }
                }

                else
                {
                    i++;
                }

                if (flag == false)
                {
                    MessageBox.Show("Invalid Credentials");
                }
            }

            return flag;
        }

    }
}
