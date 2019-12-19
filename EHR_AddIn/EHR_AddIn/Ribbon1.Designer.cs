namespace EHR_AddIn
{
    partial class Ribbon1 : Microsoft.Office.Tools.Ribbon.RibbonBase
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        public Ribbon1()
            : base(Globals.Factory.GetRibbonFactory())
        {
            InitializeComponent();
        }

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.tab1 = this.Factory.CreateRibbonTab();
            this.EHR = this.Factory.CreateRibbonGroup();
            this.Decrypt = this.Factory.CreateRibbonButton();
            this.tab1.SuspendLayout();
            this.EHR.SuspendLayout();
            this.SuspendLayout();
            // 
            // tab1
            // 
            this.tab1.Groups.Add(this.EHR);
            this.tab1.Label = "Add-Ins";
            this.tab1.Name = "tab1";
            // 
            // EHR
            // 
            this.EHR.Items.Add(this.Decrypt);
            this.EHR.Label = "EHR";
            this.EHR.Name = "EHR";
            // 
            // Decrypt
            // 
            this.Decrypt.ControlSize = Microsoft.Office.Core.RibbonControlSize.RibbonControlSizeLarge;
            this.Decrypt.Label = "Decrypt";
            this.Decrypt.Name = "Decrypt";
            this.Decrypt.ShowImage = true;
            this.Decrypt.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.Decrypt_Click);
            // 
            // Ribbon1
            // 
            this.Name = "Ribbon1";
            this.RibbonType = "Microsoft.Excel.Workbook";
            this.Tabs.Add(this.tab1);
            this.Load += new Microsoft.Office.Tools.Ribbon.RibbonUIEventHandler(this.Ribbon1_Load);
            this.tab1.ResumeLayout(false);
            this.tab1.PerformLayout();
            this.EHR.ResumeLayout(false);
            this.EHR.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        internal Microsoft.Office.Tools.Ribbon.RibbonTab tab1;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup EHR;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton Decrypt;
    }

    partial class ThisRibbonCollection
    {
        internal Ribbon1 Ribbon1
        {
            get { return this.GetRibbon<Ribbon1>(); }
        }
    }
}
